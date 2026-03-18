#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using LinearAlgebra
using Plots
using Optim
using Printf
using JLD2

# Include your existing model
include(srcdir("IonModel.jl"))
using .IonModel: build_ion_ops, ⊗, fock, Rx

# Local definition of Rz
Rz(θ) = ComplexF64[exp(-1im * θ / 2) 0; 0 exp(1im * θ / 2)]

# Helper function to build a coherent state for the testing
# |α⟩ = e^{-|α|^2/2} ∑_{n=0} ^{∞} \frac{α^n}{√n!} |n⟩
function coherent_state(α::ComplexF64, Nm::Int)
    c = zeros(ComplexF64, Nm)
    # Consider counting from 1 not 0.
    for n in 0:(Nm-1)
        c[n+1] = exp(-abs2(α) / 2) * α^n / sqrt(factorial(n))
    end
    return c
end

# Helper function for Von Neumann Entanglement Entropy 
function von_neumann_entropy(ψ::Vector{ComplexF64}, Nm::Int)
    M = reshape(ψ, Nm, 2)
    # Density matrix for a pure state; trace out the motional state
    ρ_q = M' * M 
    #  Should not be needed since it is Hermitian but not sure
    ρ_q = (ρ_q + ρ_q') / 2 
    # Take the eigenvalues of the density matric
    # This representation is possible since the density matrices are Hermitan and semi-definite
    # and therefore diagonalisable
    # The eigenvalues are the measurable probabilities of finding the system in those states
    λ = eigvals(Hermitian(ρ_q))
    # Initial 0-value for the entropy
    S = 0.0
    # Reference: Wiki Von Neumann entanglement entropy
    for val in real(λ)
        if val > 1e-15
            S -= val * log(val)
        end
    end
    return S
end

# Helper function: Phase modulation based on D4 in the Mikado paper
# Used to make the pulse smooth
function phase_mask(t::Float64, T::Float64)
    if t < 0.1 * T
        return 0.5 * (1.0 - cos(10.0 * π * t / T))
    elseif t > 0.9 * T
        return 0.5 * (1.0 + cos(10.0 * π * (t - 0.9 * T) / T))
    else
        return 1.0
    end
end

# Helper function for the Fourier coefficients
function calc_phase(t::Float64, T::Float64, a_coeffs, b_coeffs)
    val = 0.0
    Nc = length(a_coeffs) # In the paper, they suggest between 40 and 60 (so far I have not seen a reason why)
    for n in 1:Nc
        # Fourier coefficients representation
        val += a_coeffs[n] * cos(n * π * t / T) + b_coeffs[n] * sin(n * π * t / T)
    end
    # Apply the phase mask for smooth pulse.
    return val * phase_mask(t, T)
end

function optimize_and_plot()
    # Physical Regime: 88Sr Neutral Atoms - defined in Appendix I
    ω = 2π * 0.1        # 100 kHz trap frequency
    Ω0 = 2π * 0.010     # 10 kHz base Rabi freq (adjusted since they divide by 2 in their Hamiltonian)
    η = 0.22            # Lamb-Dicke parameter for Sr
    
    Nm = 6              # Motional cutoff
    T = 20.5            # Pulse duration (μs)
    Nt = 100            # Time steps for integration
    tlist = collect(range(0, T, length=Nt))
    dt = tlist[2] - tlist[1] # This is still from the Grape optimisation, but is correct here as well
    
    # Build the operattors
    ops = build_ion_ops(; Nm=Nm, ω=ω, η=η)
    
    #Build the Hamiltonian
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m
    HΔ   = ops.P1 ⊗ ops.I_m  
    
    # Split the real and the imaginary part to be able to control the phase
    HΩ_re = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
    HΩ_im = 1im * ((ops.σ10 ⊗ ops.Eplus) - (ops.σ01 ⊗ ops.Eminus))
    
    # Define these operators to be able to track and plot the state 
    n_op = ops.Iq ⊗ (ops.adag * ops.a)
    a_op = ops.Iq ⊗ ops.a
    P0_op = ops.P0 ⊗ ops.I_m
    P1_op = ops.P1 ⊗ ops.I_m

    # Thermal State Setup
    p0 = 0.90               # 90% probability of being in the ground state
    thermal_cutoff = 2      # Simulating n=0, 1, and 2 (higher cutoffs take exponentially longer, and I am currently more scanning the space)
    
    # Calculate and normalize Boltzmann weights
    p_weights = [p0 * (1 - p0)^n for n in 0:thermal_cutoff]
    p_weights ./= sum(p_weights) 

    # Base initial states for optimization
    qubit_inits = [
        ops.ket0, ops.ket1,
        (ops.ket0 + ops.ket1) / sqrt(2),
        (ops.ket0 + 1im * ops.ket1) / sqrt(2)
    ]
    
    # Build a flattened training set: (Initial State, Qubit Target Index, Thermal Weight)
    training_set = Tuple{Vector{ComplexF64}, Int, Float64}[]
    for (n_idx, w) in enumerate(p_weights)
        fock_n = n_idx - 1
        for k in 1:length(qubit_inits)
            ψ_start = qubit_inits[k] ⊗ fock(fock_n, Nm)
            push!(training_set, (ψ_start, k, w))
        end
    end

    # Defined in Appendix I
    # The intensity deviation (δI) is the fractional error in the laser's power
    # Optimise over multiple intensities to get a realistic physical pulse
    intensity_deviations = range(-0.01, 0.01, length=3)
    
    # This coefficients are a bit eyebolled; They determine the weight of the seperate cost functions.
    alpha_weight = 0.45
    ent_weight   = 0.05

    # To save the seperate cost function components.
    component_cache = Dict{Float64, Tuple{Float64, Float64, Float64}}()
    function ensemble_cost(x)
        # Remove the last 2 spots for alpha an beta, and half of the remaining are for the a_coeffs, and the other half for b_coeffs
        Nc = div(length(x) - 2, 2)
        a_coeffs = x[1:Nc]
        b_coeffs = x[Nc+1:2Nc]
        α_ang = x[end-1]
        β_ang = x[end]
        
        # Forcing the laser pulse to land on one exact, rigid mathematical matrix often means the solver gets stuck in a local minimum
        # Still need to get a bit better understanding on why for the experimentalist the z-rotations does not matter.
        Xq_target = Rz(β_ang) * Rx(π/2) * Rz(α_ang)
        # Define the initial quibit space and the corresponding target
        qubit_targets = [Xq_target * q_init for q_init in qubit_inits]
        P_targets = [(qf * qf') ⊗ ops.I_m for qf in qubit_targets]
        
        # Lists to save each component of the cost function seperatly.
        # 1. Flatten the loops into a Master List of 36 independent tasks
        N_dev = length(intensity_deviations)
        tasks = []
        for i in 1:N_dev
            for (ψ_init, k, w) in training_set
                # Save: (Intensity Index, δ_I, Initial State, Target Index, Thermal Weight)
                push!(tasks, (i, intensity_deviations[i], ψ_init, k, w))
            end
        end
        
        N_tasks = length(tasks)
        
        # 2. Allocate thread-safe arrays for ALL 36 trajectories
        task_ju = zeros(Float64, N_tasks)
        task_jm = zeros(Float64, N_tasks)
        task_je = zeros(Float64, N_tasks)
        
        # Parallelize across everything simultaneously
        Threads.@threads for s in 1:N_tasks
            i, δ_I, ψ_init, k, w = tasks[s]
            
            Ω_actual = Ω0 * sqrt(1.0 + δ_I)
            Δ_actual = 11.7 * Ω0 * δ_I 
            
            ψ = copy(ψ_init)
            
            for t in tlist[1:end-1]
                t_mid = t + dt/2
                φ_t = calc_phase(t_mid, T, a_coeffs, b_coeffs)
                H_t = H0 + Δ_actual * HΔ + (Ω_actual * cos(φ_t)) * HΩ_re + (Ω_actual * sin(φ_t)) * HΩ_im
                ψ = exp(-1im * dt * H_t) * ψ
            end
            
            fidelity = real(dot(ψ, P_targets[k] * ψ))
            heating = real(dot(ψ, n_op * ψ))
            entropy = von_neumann_entropy(ψ, Nm)
            
            # Save the weighted results for this exact trajectory
            task_ju[s] = w * (1.0 - fidelity)
            task_jm[s] = w * heating
            task_je[s] = w * entropy
        end
        
        # Reconstruct the averages per intensity deviation
        J_u_res = zeros(Float64, N_dev)
        J_m_res = zeros(Float64, N_dev)
        J_e_res = zeros(Float64, N_dev)
        
        for s in 1:N_tasks
            i = tasks[s][1] # Look up which intensity this task belonged to
            J_u_res[i] += task_ju[s] / length(qubit_inits)
            J_m_res[i] += task_jm[s] / length(qubit_inits)
            J_e_res[i] += task_je[s] / length(qubit_inits)
        end
        
        # Calculate the final averages across all intensity deviations
        total_ju = sum(J_u_res) / N_dev
        total_jm = sum(J_m_res) / N_dev
        total_je = sum(J_e_res) / N_dev
        
        # Combine them into the single cost number Optim.jl requires
        total_cost = total_ju + (alpha_weight * total_jm) + (ent_weight * total_je)
        
        # Save the separated pieces into the dictionary using the total cost as the exact key!
        component_cache[total_cost] = (total_ju, total_jm, total_je)
        return total_cost
    end


    # Optimisation
    Nc = 40 # Actually, maybe there is a difference, still have to research it
    x0 = zeros(Float64, 2*Nc + 2) 
    x0[1:2Nc] .= 0.2 .* randn(2Nc) # Increased initial kick for less stagnation
    
    println("Optimizing... Tracking Cost and Gradient history.")
    # GRAPE uses piecewise-constant control, while sines and cosines are infinitely differentiabl
    opts = Optim.Options(iterations=100, g_tol=1e-6, show_trace=true, show_every=10, store_trace=true, extended_trace=true)
    result = optimize(ensemble_cost, x0, BFGS(), opts)
    
    # Unpack the optimal results.
    x_opt = Optim.minimizer(result)
    a_opt = x_opt[1:Nc]
    b_opt = x_opt[Nc+1:2Nc]
    α_opt = x_opt[end-1]
    β_opt = x_opt[end]
    
    # --- Exact Fidelity Evaluator ---
    Xq_target_opt = Rz(β_opt) * Rx(π/2) * Rz(α_opt)
    P_targets_opt = [((Xq_target_opt * q) * (Xq_target_opt * q)') ⊗ ops.I_m for q in qubit_inits]
    
    final_fidelity = 0.0
    final_heating  = 0.0
    final_entropy  = 0.0

    for (ψ_init, k, w) in training_set
        ψ_eval = copy(ψ_init)
        for t in tlist[1:end-1]
            t_mid = t + dt/2
            φ_t = calc_phase(t_mid, T, a_opt, b_opt)
            H_t = H0 + (Ω0 * cos(φ_t)) * HΩ_re + (Ω0 * sin(φ_t)) * HΩ_im
            ψ_eval = exp(-1im * dt * H_t) * ψ_eval
        end
        
        # Calculate all three physical metrics for the ideal case
        final_fidelity += w * real(dot(ψ_eval, P_targets_opt[k] * ψ_eval))
        final_heating  += w * real(dot(ψ_eval, n_op * ψ_eval))
        final_entropy  += w * von_neumann_entropy(ψ_eval, Nm)
    end
    
    # Average over the 4 cardinal qubit states
    final_fidelity /= length(qubit_inits)
    final_heating  /= length(qubit_inits)
    final_entropy  /= length(qubit_inits)

    # print the final report
    println("Final optimised pulse report (Ideal Intensity)")
    @printf("Gate Infidelity: %.6e\n", 1.0 - final_fidelity)
    @printf("Motional Heating: %.6e phonons\n", final_heating)
    @printf("Entanglement S:   %.6e\n", final_entropy)
    
    # Post processing and plotting
    out_dir = plotsdir("MikadoReplication")
    mkpath(out_dir)

    # Simulate the trajectory for the plotting based on the optimal paramters.
    function simulate_trajectory_full(ψ_start, Ω_re_func, Ω_im_func)
        ψ = copy(ψ_start)
        x_vals, p_vals = Float64[real(dot(ψ, a_op * ψ))], Float64[imag(dot(ψ, a_op * ψ))]
        p0_vals, p1_vals = Float64[real(dot(ψ, P0_op * ψ))], Float64[real(dot(ψ, P1_op * ψ))]
        
        for t in tlist[1:end-1]
            t_mid = t + dt/2
            H_t = H0 + Ω_re_func(t_mid) * HΩ_re + Ω_im_func(t_mid) * HΩ_im
            ψ = exp(-1im * dt * H_t) * ψ
            push!(x_vals, real(dot(ψ, a_op * ψ)))
            push!(p_vals, imag(dot(ψ, a_op * ψ)))
            push!(p0_vals, real(dot(ψ, P0_op * ψ)))
            push!(p1_vals, real(dot(ψ, P1_op * ψ)))
        end
        return x_vals, p_vals, p0_vals, p1_vals
    end

    opt_re(t) = Ω0 * cos(calc_phase(t, T, a_opt, b_opt))
    opt_im(t) = Ω0 * sin(calc_phase(t, T, a_opt, b_opt))

    # Plot 1 & 2: Cost Function Deconstruction ---
    trace = Optim.trace(result)
    iters = [t.iteration for t in trace]
    costs = [t.value for t in trace] # The exact accepted costs
    grads = [t.g_norm for t in trace]

    println("Instantly retrieving cost components from cache...")
    J_uni_history = Float64[]
    J_mot_history = Float64[]
    J_ent_history = Float64[]

    # Retrive the seperate cost components.
    for c in costs
        ju, jm, je = component_cache[c]
        push!(J_uni_history, ju)
        push!(J_mot_history, jm)
        push!(J_ent_history, je)
    end

    # Panel 1: Stacked Cost History
    p_cost = plot(iters, costs, yaxis=:log10, label="Total Cost", lw=3, color=:black, 
                  xlabel="Iteration", ylabel="Magnitude (Log)", title="Optimization Trajectory", grid=true)
    
    plot!(p_cost, iters, J_uni_history, label="Infidelity (J_uni)", lw=2, color=:dodgerblue)
    plot!(p_cost, iters, alpha_weight .* J_mot_history, label="Heating Penalty", lw=2, color=:orange)
    plot!(p_cost, iters, ent_weight .* J_ent_history, label="Entropy Penalty", lw=2, color=:purple)
    plot!(p_cost, iters, grads, label="Gradient Norm", lw=1.5, color=:gray, linestyle=:dash)
    # Panel 2: Qubit Populations
    q_fig2 = sin(π/8) * ops.ket0 + 1im * cos(π/8) * ops.ket1
    ψ_left_start = q_fig2 ⊗ fock(0, Nm)
    
    _, _, p0_opt_L, p1_opt_L = simulate_trajectory_full(ψ_left_start, opt_re, opt_im)
    
    p_qubit = plot(tlist, p0_opt_L, label="P(|0⟩)", lw=2.5, color=:dodgerblue, 
                   xlabel="Time (μs)", ylabel="Population", 
                   title=@sprintf("Qubit State (Final Gate Fidelity: %.6f)", final_fidelity), grid=true)
    plot!(p_qubit, tlist, p1_opt_L, label="P(|1⟩)", lw=2.5, color=:crimson, linestyle=:dash)
    
    fig_cost_qubit = plot(p_cost, p_qubit, layout=(1,2), size=(1000, 450), margin=5Plots.mm)
    display(fig_cost_qubit)
    savefig(fig_cost_qubit, joinpath(out_dir, "cost_and_qubit_state T=$T eta=$η Nt=$Nt dev = 3 init_kick=0.2  Nc=$Nc  alpha_weight=$alpha_weight ent_weight=$ent_weight.png"))

    # Phase space
    Ω_MB = π / (4 * T) 
    mb_re(t) = Ω_MB
    mb_im(t) = 0.0

    x_opt_L, p_opt_L, _, _ = simulate_trajectory_full(ψ_left_start, opt_re, opt_im)
    x_mb_L,  p_mb_L, _, _  = simulate_trajectory_full(ψ_left_start, mb_re, mb_im)
    
    p_left = plot(x_opt_L, p_opt_L, title="Initial |0⟩m", aspect_ratio=:equal, lw=2, color=:purple, label="Recoil-free")
    plot!(p_left, x_mb_L, p_mb_L, lw=2, color=:orange, linestyle=:dash, label="Mößbauer")
    scatter!(p_left, [x_opt_L[1]], [p_opt_L[1]], color=:green, markersize=5, label="Start/End (Ideal)")
    xlabel!(p_left, "⟨x⟩ / (2x₀)")
    ylabel!(p_left, "⟨p⟩ / (2p₀)")

    α0 = 0.15 + 0.0im
    ψ_right_start = q_fig2 ⊗ coherent_state(α0, Nm)
    
    x_opt_R, p_opt_R, _, _ = simulate_trajectory_full(ψ_right_start, opt_re, opt_im)
    
    θ_circ = range(0, 2π, length=100)
    circ_x = abs(α0) .* cos.(θ_circ)
    circ_y = abs(α0) .* sin.(θ_circ)
    
    p_right = plot(circ_x, circ_y, title="Initial |α=0.15⟩m", aspect_ratio=:equal, lw=1, color=:gray, linestyle=:dot, label="Reference Circle")
    plot!(p_right, x_opt_R, p_opt_R, lw=2, color=:purple, label="Recoil-free")
    scatter!(p_right, [x_opt_R[1]], [p_opt_R[1]], color=:green, markersize=5, label="Start")
    scatter!(p_right, [x_opt_R[end]], [p_opt_R[end]], color=:red, markersize=5, label="End")
    xlabel!(p_right, "⟨x⟩ / (2x₀)")
    
    fig2 = plot(p_left, p_right, layout=(1,2), size=(900, 450), grid=true)
    display(fig2)
    savefig(fig2, joinpath(out_dir, "figure_2_phase_space T=$T eta=$η Nt=$Nt  Nc=$Nc dev =3 init_kick=0.2  alpha_weight=$alpha_weight ent_weight=$ent_weight.png"))

    println("All plots successfully generated and saved to: $out_dir")

    # Save the optimised pulse
    filename = "optimized_pulse_T=$T eta=$η Nt=$Nt  Nc=$Nc dev=3 init_kick=0.2  alpha_weight=$alpha_weight ent_weight=$ent_weight.jld2"
    save_path = datadir("sims", filename)
    mkpath(datadir("sims")) # Ensure the folder exists
    
    @save save_path x_opt a_opt b_opt α_opt β_opt final_fidelity
    println("Optimised pulse saved to: $save_path")
end

optimize_and_plot()