#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using ArgParse
using LinearAlgebra
using Random
using Printf
using JLD2
using Plots

include(srcdir("IonModel.jl"))
using .IonModel: build_ion_ops, Rx, fock, ⊗

const ns = 1.0

"""Bell state |Φ+> on (Reference ⊗ System) as a 4-vector."""
# Reference is a perfect, stattionary quibit that does not interact with any laser or the environment
# System is the actual qubit that is subject tot the lasr Pulses
# The goal is to see id the gate applied to S preserves the entanglement with R
function bell_phi_plus()
    ψ = zeros(ComplexF64, 4)
    ψ[1] = 1 / sqrt(2)  # |00>
    ψ[4] = 1 / sqrt(2)  # |11>
    return ψ
end

"""Partial trace over motion: (RS ⊗ m) -> RS density matrix."""
function partial_trace_motion_RS(ψ::Vector{ComplexF64}, Nm::Int)
    # Reshape: rows = RS (dim 4), cols = motion (Nm)
    M = reshape(ψ, Nm, 4)      
    ρ = M' * M                 # Tr_m(|ψ><ψ|)
    ρ = (ρ + ρ') / 2           # Enforce Hermitian
    ρ ./= real(tr(ρ))          # Normalise
    return Hermitian(ρ)
end


# clean up small numerical errors like a tiny negative number,
function sanitize_psd(ρ; ϵ=1e-12, normalize::Bool=true)
    ρh = Hermitian((ρ + ρ') / 2)
    vals, vecs = eigen(ρh)
    # forces the negative numbers back to 0
    vals = clamp.(real(vals), 0.0, Inf) 
    ρpsd = vecs * Diagonal(vals) * vecs'
    
    # ensures that the Trace is exactly 1.0
    if normalize
        trρ = real(tr(ρpsd))
        return (trρ > ϵ) ? Hermitian(ρpsd / trρ) : Hermitian(ρpsd)
    else
        return Hermitian(ρpsd)
    end
end

# Calculate the Quantum Fidelity based on Nielsen and ...
function sqrt_psd(ρ)
    ρ = sanitize_psd(ρ; normalize=true)
    vals, vecs = eigen(ρ)

    # Since ρ is Hermitian, we can calculate the square root by taking the square root of its eigenvalues: √ρ = V √Λ V^† .
    return vecs * Diagonal(sqrt.(clamp.(real(vals), 0.0, Inf))) * vecs'
end

"""Calculate the fidelity between two density matrices ρ and σ."""
# Use Uhlmann-Jozsa fidelity for mixed quantum states - provides a measure of how similar the two density states are 
# ρ is the density matrix of the reference-system quibit after the simulation has run
# It is obtained by taking the full wavefunction Reference-System-Motional and  performing a partial trace to remove the motional
# σ is the perfect density matrix we want to achieve; for an X gate, this is the state where the System qubit has been rotated
# exactly by θ while remaining perfectly entangled with the Reference qubit.
function fidelity_density(ρ::AbstractMatrix{ComplexF64}, σ::AbstractMatrix{ComplexF64})
    ρ = sanitize_psd(ρ; normalize=true)
    σ = sanitize_psd(σ; normalize=true)
    sρ = sqrt_psd(ρ)
    term = sanitize_psd(sρ * σ * sρ; normalize=false)
    vals = eigen(term).values
    F = (sum(sqrt.(clamp.(real(vals), 0.0, Inf))))^2
    return clamp(F, 0.0, 1.0)
end


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

function entanglement_curve_refqubit(ops, Ω_vals::AbstractVector, Δ_vals::AbstractVector, tlist; θ=π, ψm::Vector{ComplexF64})
    Nm = ops.Nm
    I2 = Matrix{ComplexF64}(I, 2, 2)

    # Hamiltonian pieces for (System ⊗ Motion)
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m
    HΔ   = ops.P1 ⊗ ops.I_m
    HΩ   = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)

    # Lift to (Reference ⊗ System ⊗ Motion)
    H0_big = I2 ⊗ H0
    HΔ_big = I2 ⊗ HΔ
    HΩ_big = I2 ⊗ HΩ

    # Initial state: Bell state on RS, fock state on m
    ψRS = bell_phi_plus()
    ψ   = ψRS ⊗ ψm

    # Ideal target: Apply X gate to the System qubit of the Bell state
    X = Rx(θ)
    U_RS_ideal = I2 ⊗ X
    ψRS_ideal  = U_RS_ideal * ψRS
    ρ_RS_ideal = sanitize_psd(ψRS_ideal * ψRS_ideal')

    Nt = length(tlist)
    F = Vector{Float64}(undef, Nt)

    # Initial fidelity at t=0
    ρ_RS = partial_trace_motion_RS(ψ, Nm)
    F[1] = fidelity_density(Matrix(ρ_RS), Matrix(ρ_RS_ideal))

    # Evolution loop
    for k in 1:Nt-1
        Δt = tlist[k+1] - tlist[k]
        Hk = H0_big + Ω_vals[k] * HΩ_big
        if !isempty(Δ_vals)
            Hk += Δ_vals[k] * HΔ_big
        end
        ψ  = exp(-1im * Δt * Hk) * ψ
        ρ_RS = partial_trace_motion_RS(ψ, Nm)
        F[k+1] = fidelity_density(Matrix(ρ_RS), Matrix(ρ_RS_ideal))
    end
    return F
end

function thermal_probs(Nm::Int; nbar::Real)
    q = nbar / (1 + nbar)
    p = [(1 - q) * q^n for n in 0:Nm-1]
    return p ./ sum(p)
end


function main()
    args = parse_commandline()
    sims_dir = datadir("sims")
    
    # 1. Load File
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

    # 2. Extract Params & Controls
    Nm, ω, η, T, dt = payload["Nm"], payload["ω"], payload["η"], payload["T"], payload["dt"]
    θ = π
    tlist = collect(0:dt:T)
    t_ns  = tlist ./ ns

    history = payload["history"]
    last_step = history[end]
    Ω_opt_vals = last_step.Ω   
    
    params = get(payload, "params", Dict("mode" => "full"))
    mode = get(params, "mode", "full")
    Δ_opt_vals = (mode == "full") ? last_step.Δ : Float64[]

    ops = build_ion_ops(; Nm=Nm, ω=ω, η=η)
    nbar = get(payload, "nbar", 0.2)
    p_th = thermal_probs(Nm; nbar=nbar)

    # 3. Setup Plots
    out_dir = plotsdir("Entanglement")
    mkpath(out_dir)

    p = plot(title="Entanglement Fidelity (Ref Qubit) - nbar=$nbar",
             xlabel="Time (ns)", ylabel="Fidelity",
             legend=:bottomright, ylims=(0, 1.05))

    F_per_n = zeros(Float64, Nm)

    # 4. Run Tests per Fock state
    println("Running entanglement-fidelity curves for each Fock n...")
    for n in 0:Nm-1
        ψm = fock(n, Nm)
        curve = entanglement_curve_refqubit(ops, Ω_opt_vals, Δ_opt_vals, tlist; θ=θ, ψm=ψm)
        F_per_n[n+1] = curve[end]
        plot!(p, t_ns, curve, label="n=$n (p=$(round(p_th[n+1]; sigdigits=3)))", lw=2)
    end

    # 5. Final Metrics
    F_weighted = sum(p_th .* F_per_n)
    F_avg = (2 * F_weighted + 1) / 3

    Printf.@printf("\nThermal-weighted entanglement fidelity F_e(T) = %.8f\n", F_weighted)
    Printf.@printf("Implied average gate fidelity F_avg ≈ %.8f\n", F_avg)
    
    # 6. Save Result
    name_tag = replace(basename(target_file), ".jld2" => "")
    out_path = joinpath(out_dir, "$(name_tag)_EntFid.png")
    savefig(p, out_path)
    println("Saved plot to: ", out_path)
end

main()