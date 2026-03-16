#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using LinearAlgebra
using Random
using Printf
using JLD2
using DataFrames
using Plots
using Logging

# GRAPE & QuantumControl imports exactly as in your script
using GRAPE
using QuantumControl
using QuantumControl: Trajectory, propagate_trajectories
using QuantumPropagators
using QuantumPropagators.Amplitudes: ShapedAmplitude
using QuantumPropagators.Shapes: flattop
using QuantumPropagators: Cheby
using Optim
using LineSearches

# Ensure we load modules from the src/ directory
include(srcdir("Types.jl"))
include(srcdir("IonModel.jl"))
include(srcdir("Utils.jl"))

using .IonModel: build_ion_ops, ion_hamiltonian_ΩΔ, Rx, fock, ⊗

const GHz = 1.0
const MHz = 0.001GHz
const ns  = 1.0
const WALL_WEIGHT = 10000.0

const OMEGA_0_VAL = 35 * 2π * MHz
rectshape(t::Float64)::Float64 = 1.0
ϵ_Ω_guess_global(t::Float64)::Float64 = OMEGA_0_VAL

"""
Calculates the Von Neumann Entanglement Entropy S = -Tr(ρ_q * ln(ρ_q)).
"""
function von_neumann_entropy(ψ::Vector{ComplexF64}, Nm::Int)
    M = reshape(ψ, Nm, 2)
    ρ_q = M' * M 
    ρ_q = (ρ_q + ρ_q') / 2 # Ensure strictly Hermitian
    
    λ = eigvals(Hermitian(ρ_q))
    
    S = 0.0
    for val in real(λ)
        if val > 1e-15
            S -= val * log(val)
        end
    end
    return S
end

function thermal_probs(Nm::Int; nbar::Real)::Vector{Float64}
    q = nbar / (1 + nbar)                 
    p = [(1 - q) * q^n for n in 0:Nm-1]    
    s = sum(p)
    return p ./ s
end

function haar_state(d::Int; rng=Random.default_rng())
        v = randn(rng, ComplexF64, d)
        return v / norm(v)
end

function optimize_and_track(T_ns, η, Δ_MHz; Nm=12, maxiters=500, nbar=0.2, seed=1234)
    T = T_ns * ns
    ω_trap = 2π * 2.5 * MHz
    Δ_rad = Δ_MHz * 2π * MHz
    θ = π
    Ωmax = 500 * 2π * MHz
    
    ops = build_ion_ops(; Nm=Nm, ω=ω_trap, η=η)

    # 1. Setup the full ensemble of initial and target states
    rng = MersenneTwister(seed)
    ψq1 = haar_state(2; rng=rng)
    ψq2 = haar_state(2; rng=rng)

    qubit_inputs = [ops.ket0, ops.ket1, ψq1, ψq2]
    Xq = Rx(θ)
    qubit_targets = [Xq * q for q in qubit_inputs]
    P_targets = [(qf*qf') ⊗ ops.I_m for qf in qubit_targets]

    # 2. Fix for GRAPE tlist length: Must be >= 3. 
    tlist = collect(range(0, T, length=3))
    nseg = 2
    
    # 3. Hamiltonian setup
    Ω_amp = ShapedAmplitude(ϵ_Ω_guess_global; shape=rectshape)
    
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m
    HΔ   = ops.P1 ⊗ ops.I_m
    H_drift = H0 + Δ_rad * HΔ
    HΩ   = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
    
    Hgen = QuantumPropagators.hamiltonian(H_drift, (HΩ, Ω_amp))

    # 4. Build Trajectories for the thermal ensemble
    p_th = thermal_probs(Nm; nbar=nbar)
    trajectories = Trajectory[]
    weights = Float64[]

    for n in 0:Nm-1
        w = p_th[n+1]
        ψm = fock(n, Nm)
        push!(trajectories, QuantumControl.Trajectory(qubit_inputs[1] ⊗ ψm, Hgen; P_target=P_targets[1]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(qubit_inputs[2] ⊗ ψm, Hgen; P_target=P_targets[2]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(qubit_inputs[3] ⊗ ψm, Hgen; P_target=P_targets[3]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(qubit_inputs[4] ⊗ ψm, Hgen; P_target=P_targets[4]))
        push!(weights, w)
    end

    # --- EXACT WORKSPACES FROM YOUR CODE ---
    state_length = 2 * Nm 
    num_traj = length(trajectories)
    workspace_JT = [zeros(ComplexF64, state_length) for _ in 1:num_traj]
    num_pulse_vals = nseg
    workspace_grad = zeros(Float64, num_pulse_vals)
    workspace_chi = [zeros(ComplexF64, state_length) for _ in 1:num_traj]

    # --- EXACT FUNCTIONS FROM YOUR CODE ---
    function J_T_qubitproj(Ψ, trajectories) 
        s = 0.0
        wsum = sum(weights)
        for k in eachindex(Ψ)
            mul!(workspace_JT[k], trajectories[k].P_target, Ψ[k])
            s += weights[k] * real(dot(Ψ[k], workspace_JT[k]))
        end
        return 1 - s / wsum
    end
 
    function chi_qubitproj(Ψ, trajectories)
        wsum = sum(weights)
        for k in eachindex(Ψ)
            mul!(workspace_chi[k], trajectories[k].P_target, Ψ[k])
            workspace_chi[k] .*= (weights[k] / wsum) 
        end
        return workspace_chi
    end

    function J_slew(pulsevals, tlist)  
        return 10000.0 * (pulsevals[2] - pulsevals[1])^2
    end

    function J_boundary(pulsevals, tlist)
        penalty = 0.0
        for i in 1:nseg
            v = pulsevals[i]
            if v < 0.0
                penalty += v^4
            elseif v > Ωmax
                penalty += (v - Ωmax)^4
            end
        end
        return WALL_WEIGHT * penalty
    end

    function J_total_penalty(pulsevals, tlist) 
        return J_boundary(pulsevals, tlist) + J_slew(pulsevals, tlist)
    end
        
    function grad_J_total_penalty(pulsevals, tlist)
        fill!(workspace_grad, 0.0)
        for i in 1:nseg
            v = pulsevals[i]
            if v < 0.0
                workspace_grad[i] += WALL_WEIGHT * 4 * v^3
            elseif v > Ωmax
                workspace_grad[i] += WALL_WEIGHT * 4 * (v - Ωmax)^3
            end
        end
        diff = pulsevals[2] - pulsevals[1]
        workspace_grad[1] += -20000.0 * diff
        workspace_grad[2] +=  20000.0 * diff
        return workspace_grad
    end

    # 5. Run GRAPE Optimization
    result = GRAPE.optimize(
        trajectories, tlist;
        use_threads=true,
        J_T = J_T_qubitproj,
        J_a = J_total_penalty,
        grad_J_a = grad_J_total_penalty,
        chi = chi_qubitproj,
        prop_method = Cheby,
        maxiters = maxiters,
        show_trace = false, 
        print_iters = false,
        pulse_options = Dict(),  
        optimizer = Optim.BFGS(linesearch = LineSearches.BackTracking()), 
    )

    Ω_opt_val = (result.optimized_controls[1][1] + result.optimized_controls[1][2]) / 2.0

    # 6. Track Entanglement Entropy fine-grained over time (Test on |0,0> state)
    H_opt = H_drift + Ω_opt_val * HΩ
    track_dt = T / 200
    track_tlist = collect(0:track_dt:T)
    S_t = zeros(Float64, length(track_tlist))
    
    # We strictly test the optimized pulse on the |0> ⊗ |0> state for the entropy graph
    # ψ_t = ops.ket0 ⊗ fock(0, Nm) 
    # state = 1/sqrt(2) * (|0⟩ ⊗ |n=0⟩ + |1⟩ ⊗ |n=1⟩)
    term1 = ops.ket0 ⊗ fock(4, Nm)
    term2 = ops.ket1 ⊗ fock(6, Nm)
    ψ_t = (term1 + term2) / sqrt(2.0)
    U_dt = exp(-1im * track_dt * H_opt)
    
    P_mn_t = zeros(Float64, Nm, length(track_tlist))

    for (i, t) in enumerate(track_tlist)
        S_t[i] = von_neumann_entropy(ψ_t, Nm)
        M_reshape = reshape(ψ_t, Nm, 2)
        P_mn_t[:, i] = vec(sum(abs2, M_reshape; dims=2))
        if i < length(track_tlist)
            ψ_t = U_dt * ψ_t
        end
    end
    
    max_S = maximum(S_t)

    return result.J_T, max_S, Ω_opt_val, track_tlist, S_t, P_mn_t
end

function main()
    # Your requested parameter grid
    println("Threads available: ", Threads.nthreads())
    Ts = [10.0, 20.0, 40.0, 50.0]
    etas = [0.1, 0.5, 0.8]
    detunings = [0.0]
    Nm = 12 
    
    results = []
    
    # Create directory for the individual trajectory plots
    indiv_plots_dir = plotsdir("pulse_and_entropy_trajectories")
    mkpath(indiv_plots_dir)

    println("Starting Parameter Sweep utilizing your exact GRAPE infrastructure...")
    
    for T in Ts, η in etas, Δ in detunings
        @printf("Optimizing: T = %5.1fns, η = %.1f, Δ = %.1fMHz...", T, η, Δ)
        
        # Run the optimization and extract the time-series arrays
        J_T, S_max, Ω_opt, t_array, S_array, P_mn_array = optimize_and_track(T, η, Δ; Nm=Nm)
        
        @printf(" J_T = %.2e, S_max = %.4f\n", J_T, S_max)
        
        # Save to our collection (includes the individual optimal control and the arrays)
        push!(results, (
            T = T, 
            η = η, 
            Δ = Δ, 
            J_T = J_T, 
            S_max = S_max, 
            Ω_opt = Ω_opt, 
            tlist = t_array, 
            S_t = S_array
        ))
        
        # Generate the combined Control Pulse & Entropy plot
        Ω_MHz = Ω_opt / (2π * MHz)
        Ω_array = fill(Ω_MHz, length(t_array)) # Create a flat array for the constant pulse
        
        # 1. Top Panel: Control Graph
        p_ctrl = plot(t_array, Ω_array, 
            ylabel = "Ω (2π·MHz)", 
            title = "T=$(T)ns, η=$η, Δ=$(Δ)MHz",
            lw = 3, 
            legend = false,
            color = :dodgerblue,
            grid = true,
            ylims = (0, max(Ω_MHz * 1.5, 5.0)) # Add some padding above the line
        )
        
        # 2. Bottom Panel: Entropy Graph
        p_ent = plot(t_array, S_array, 
            xlabel = "Time (ns)", 
            ylabel = "Entanglement Entropy (S)",
            lw = 3, 
            legend = false,
            color = :indigo,
            grid = true,
            ylims = (0, max(S_max * 1.2, 0.1))
        )
        
        # 3. Third Panel: Motional Heatmap
        p_motion = heatmap(t_array, 0:Nm-1, P_mn_array,
            xlabel = "Time (ns)", 
            ylabel = "Fock state n",
            title = "Motional Dist p(n,t)",
            color = :viridis,
            colorbar = false
        )
        
        # Stack all three vertically
        p_combined = plot(p_ctrl, p_ent, p_motion, layout=(3,1), size=(700, 900), margin=5Plots.mm)
        
        plot_name = @sprintf("pulse_and_entropy_T%.0f_eta%.1f_Delta%.1f.png", T, η, Δ)
        savefig(p_combined, joinpath(indiv_plots_dir, plot_name))
        
        
    end

    df = DataFrame(results)
    
    # 1. Save all Data (including the optimal controls and time arrays)
    save_path = datadir("sims", "entanglement_entropy_grape_exact.jld2")
    mkpath(dirname(save_path))
    wsave(save_path, Dict("df" => df))
    println("\nData saved successfully to $save_path")
    println("Individual pulse and entropy trajectory plots saved to $indiv_plots_dir")
    
    # 2. Plotting the aggregate macro-result (Final Cost vs Max Entropy)
    p_macro = scatter(df.S_max, df.J_T, 
        group = string.("η=", df.η, ", Δ=", df.Δ),
        xlabel="Max Entanglement Entropy (S)", 
        ylabel="Final Gate Infidelity (J_T)",
        yscale=:log10,
        title="Exact GRAPE Const Ω: Performance vs Peak Entanglement",
        marker=:circle,
        legend=:outertopright,
        size=(850, 550),
        dpi=300,
        grid=true,
        margin=5Plots.mm
    )
    
    plot_path = plotsdir("entanglement_vs_infidelity_exact.png")
    savefig(p_macro, plot_path)
    println("Aggregate plot saved successfully to $plot_path")

    unique_Ts = sort(unique(df.T))
    unique_etas = sort(unique(df.η))
    unique_deltas = sort(unique(df.Δ))
       
    println("Generating S_max temperature heatmaps...")
        
    for T_val in unique_Ts
        df_T = filter(row -> row.T == T_val, df)
            
        # Initialize a 2D matrix (rows = Δ, columns = η)
        Z = zeros(Float64, length(unique_deltas), length(unique_etas))
            
        for (i, d) in enumerate(unique_deltas)
            for (j, e) in enumerate(unique_etas)
                    # Find the matching run
                row = filter(r -> r.η == e && r.Δ == d, df_T)
                if nrow(row) > 0
                    Z[i, j] = row.S_max[1]
                else
                    Z[i, j] = NaN # Handle missing data gracefully
                end
            end
        end
            
        p_heat = heatmap(unique_etas, unique_deltas, Z, 
            xlabel="Lamb-Dicke Parameter (η)", 
            ylabel="Detuning Δ (MHz)", 
            title="Max Entanglement S_max (T = $(T_val)ns)",
            color=:plasma, 
            colorbar_title="S_max",
            size=(600, 500),
            margin=5Plots.mm
            )
            
        heat_path = plotsdir("Smax_heatmap_T$(T_val).png")
        savefig(p_heat, heat_path)
    end
        
    println("All plots completed.")
end

main()
