#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using ArgParse
using LinearAlgebra
using Plots
using JLD2

include(srcdir("IonModel.jl"))
using .IonModel: build_ion_ops, fock, Rx, ⊗

# ARGUMENT PARSER 
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--file"
            help = "Specific .jld2 filename in data/sims/. If left blank, gets the most recent."
            arg_type = String
            default = ""
    end
    return parse_args(s)
end

function partial_trace_qubit(ψ::Vector{ComplexF64}, Nm::Int)
    M = reshape(ψ, Nm, 2) 
    ρ_m = M * M'                 
    ρ_m = (ρ_m + ρ_m') / 2       
    return Hermitian(ρ_m)
end

# estimate the robustness and long-term stability of the optimised pulse
function error_amplification_curve(ops, Ω_vals, Δ_vals, tlist, max_repeats::Int, n_fock::Int; θ=π)
    Nm = ops.Nm
    Nt = length(tlist)
    dim = 2 * Nm
    U_gate = Matrix{ComplexF64}(I, dim, dim)
    
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m
    HΔ   = ops.P1 ⊗ ops.I_m
    HΩ   = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
    
    # Handle omega_only mode where Δ_vals might be empty
    for k in 1:Nt-1
        # discretise the contninous function
        Δt = tlist[k+1] - tlist[k]
        Hk = H0 + Ω_vals[k] * HΩ
        if !isempty(Δ_vals)
            Hk += Δ_vals[k] * HΔ
        end
        # calculates the propagator for delta t
        Uk = exp(-1im * Δt * Hk)
        # by the time when the time loop finished, U_gate represent the entire effect of the pulse from start to finish
        U_gate = Uk * U_gate
    end
    
    ψ_current = ops.ket0 ⊗ fock(n_fock, Nm)
    ψq_ideal = copy(ops.ket0) 
    X_gate = Rx(θ)            
    
    infidelities = Float64[]
    mean_phonons = Float64[]

    # loop ove the mutliple gates
    for N in 1:max_repeats
        ψ_current = U_gate * ψ_current
        ψq_ideal = X_gate * ψq_ideal
        P_target = (ψq_ideal * ψq_ideal') ⊗ ops.I_m
        
        overlap = real(dot(ψ_current, P_target * ψ_current))
        J_T = 1.0 - overlap
        push!(infidelities, J_T)

        ρ_m = partial_trace_qubit(ψ_current, Nm)
        P_n = real.(diag(ρ_m))
        n_avg = sum((0:Nm-1) .* P_n)
        push!(mean_phonons, n_avg)
    end
    
    return infidelities, mean_phonons
end

function main()
    args = parse_commandline()
    sims_dir = datadir("sims")
    
    if args["file"] != ""
        target_file = joinpath(sims_dir, args["file"])
        isfile(target_file) || error("File not found: $(args["file"])")
    else
        files = filter(f -> endswith(f, ".jld2"), readdir(sims_dir; join=true))
        isempty(files) && error("No .jld2 files found in $sims_dir")
        mtimes = stat.(files) .|> s -> s.mtime
        target_file = files[argmax(mtimes)]
    end
    
    println("Loading: ", basename(target_file))
    payload = JLD2.load(target_file, "payload")
    
    Nm, ω, η, T, dt = payload["Nm"], payload["ω"], payload["η"], payload["T"], payload["dt"]
    tlist = collect(0:dt:T)
    
    last_step = payload["history"][end]
    Ω_opt_vals = last_step.Ω
    
    # Graceful fallback for omega_only mode
    params = get(payload, "params", Dict("mode" => "full"))
    Δ_opt_vals = get(params, "mode", "full") == "full" ? last_step.Δ : Float64[]

    ops = build_ion_ops(; Nm=Nm, ω=ω, η=η)
    
    max_repeats = 21
    n_fock = 0.2       
    θ = π 
    
    println("Running error amplification up to $max_repeats gates...")
    infidelities, mean_phonons = error_amplification_curve(ops, Ω_opt_vals, Δ_opt_vals, tlist, max_repeats, n_fock; θ=θ)
    
    repeats_list = 1:max_repeats
    p = plot(layout=(2,1), size=(800,600), legend=:topleft, margin=5Plots.mm)
    
    plot!(p[1], repeats_list, infidelities, marker=:circle, color=:blue, 
          label="J_T (Infidelity)", ylabel="Infidelity", 
          title="Error Amplification (Initial n=$n_fock)", yscale=:log10)
          
    plot!(p[2], repeats_list, mean_phonons, marker=:diamond, color=:green, 
          label="<n>", ylabel="Mean Phonons", xlabel="Gate Applications (N)")
    
    # Save using DrWatson's plotsdir
    name_tag = replace(basename(target_file), ".jld2" => "")
    out_dir = plotsdir("ErrorAmplification")
    mkpath(out_dir)
    
    out_path = joinpath(out_dir, "$(name_tag)_ErrorAmp.png")
    savefig(p, out_path)
    println("Plot saved successfully to: ", out_path)
end

main()