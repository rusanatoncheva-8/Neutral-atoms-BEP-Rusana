#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using LinearAlgebra
using Plots
using QuantumControl
using QuantumPropagators
using GRAPE

# Include your existing model exactly as it is written
include(srcdir("IonModel.jl"))
using .IonModel: build_ion_ops, ⊗, fock, Rx

function compare_phase_space()
    # --- 1. System Parameters ---
    η = 0.5       
    T = 2π         
    Ω_const = 0.25 
    
    # Force EXACTLY 201 points 
    Nt_points = 201 
    tlist = collect(range(0, T, length=Nt_points))
    dt = tlist[2] - tlist[1] 
    Nt = length(tlist) 
    
    ω = 0.025 
    Nm = 15 
    
    println("\n--- Processing trap frequency ω = $ω (η = $η) ---")

    # --- 2. Setup Operators ---
    ops = build_ion_ops(; Nm=Nm, ω=ω, η=η)
    
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m
    
    HΩ_re = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
    HΩ_im = 1im * ((ops.σ10 ⊗ ops.Eplus) - (ops.σ01 ⊗ ops.Eminus))
    
    n_op = ops.Iq ⊗ (ops.adag * ops.a)
    
    # --- 3. GRAPE Optimization ---
    println("Optimizing GRAPE pulse with Multi-Objective Cost & Penalties...")
    
    # 1. Seed the phase to break symmetry using discrete time bins
    # GRAPE controls live on the intervals between time points, so length is Nt-1
    t_pulse = tlist[1:Nt-1] 
    phi_guess = [0.1 * sin(2 * π * t / T) for t in t_pulse]
    
    # 2. Map this phase onto Cartesian control ARRAYS (Vector{Float64})
    Ω_re_guess = [Ω_const * cos(val) for val in phi_guess]
    Ω_im_guess = [Ω_const * sin(val) for val in phi_guess]
    
    # 3. Pass the raw arrays directly to the Hamiltonian! 
    # Do not use ShapedAmplitude wrappers; GRAPE needs the raw arrays to mutate them.
    Hgen = QuantumPropagators.hamiltonian(H0, (HΩ_re, Ω_re_guess), (HΩ_im, Ω_im_guess))
    
    # --- THE 4 TRAINING STATES ---
    Xq_target = Rx(π/2)
    qubit_inits = [
        ops.ket0,
        ops.ket1,
        (ops.ket0 + ops.ket1) / sqrt(2),
        (ops.ket0 + 1im * ops.ket1) / sqrt(2)
    ]
    
    qubit_targets = [Xq_target * q_init for q_init in qubit_inits]
    ψ_inits = [q_init ⊗ fock(0, Nm) for q_init in qubit_inits]
    P_targets = [(qf * qf') ⊗ ops.I_m for qf in qubit_targets]
    
    trajectories = [
        Trajectory(ψ_inits[k], Hgen; P_target=P_targets[k]) 
        for k in 1:length(ψ_inits)
    ]
    
    # --- WORKSPACES & CUSTOM TARGET COST FUNCTIONS ---
    state_length = length(ψ_inits[1])
    num_traj = length(trajectories)
    
    workspace_chi = [zeros(ComplexF64, state_length) for _ in 1:num_traj]
    workspace_n   = [zeros(ComplexF64, state_length) for _ in 1:num_traj]
    
    # Pre-allocate gradient arrays for out-of-place return
    workspace_grad_re = zeros(Float64, Nt-1)
    workspace_grad_im = zeros(Float64, Nt-1)
    
    alpha_weight = 0.1 
    
    function custom_J_T(Ψ, trajectories; kwargs...)
        J = 0.0
        N = length(Ψ)
        for k in 1:N
            P_tgt = trajectories[k].P_target 
            fidelity = real(dot(Ψ[k], P_tgt * Ψ[k]))
            heating = real(dot(Ψ[k], n_op * Ψ[k]))
            J += (1.0 - fidelity) + (alpha_weight * heating)
        end
        return J / N
    end

    # CORRECTED STATE GRADIENT (chi)
    function custom_chi(Ψ, trajectories; kwargs...)
        N = length(Ψ)
        for k in 1:N
            P_tgt = trajectories[k].P_target
            mul!(workspace_chi[k], P_tgt, Ψ[k])
            mul!(workspace_n[k], n_op, Ψ[k])
            
            # |chi> = (P_tgt * Ψ - alpha * n_op * Ψ) / N
            workspace_chi[k] .= (workspace_chi[k] .- alpha_weight .* workspace_n[k]) ./ N
        end
        # CRITICAL: Return copies so GRAPE's line search doesn't corrupt the history
        return [copy(v) for v in workspace_chi]
    end

    # --- PENALTY FUNCTIONS (Amplitude & Slew) ---
    AMP_WEIGHT = 10.0     # Lowered to allow the optimizer to take initial steps
    SLEW_WEIGHT = 5.0     # Keeps the phase from oscillating wildly
    Ω_target = Ω_const    # 0.25

    function J_a_penalty(pulses, tlist)
        penalty = 0.0
        Ω_re = pulses[1]
        Ω_im = pulses[2]
        
        # 1. Constant Amplitude Penalty: (Re^2 + Im^2 - Ω_target^2)^2
        for i in 1:length(Ω_re)
            amp_sq = Ω_re[i]^2 + Ω_im[i]^2
            penalty += AMP_WEIGHT * (amp_sq - Ω_target^2)^2
        end
        
        # 2. Slew Penalty
        for pulsevals in pulses
            for i in 1:(length(pulsevals)-1)
                diff = pulsevals[i+1] - pulsevals[i]
                penalty += SLEW_WEIGHT * diff^2
            end
        end
        
        return penalty
    end

    # CORRECTED PENALTY GRADIENT (grad_J_a)
    # Uses 2-argument signature and returns pre-allocated arrays
    function custom_grad_J_a(pulses, tlist)
        Ω_re = pulses[1]
        Ω_im = pulses[2]
        
        fill!(workspace_grad_re, 0.0)
        fill!(workspace_grad_im, 0.0)
        
        # 1. Amplitude Penalty Gradients
        for i in 1:length(Ω_re)
            amp_sq = Ω_re[i]^2 + Ω_im[i]^2
            diff_sq = amp_sq - Ω_target^2
            
            workspace_grad_re[i] += 4.0 * AMP_WEIGHT * Ω_re[i] * diff_sq
            workspace_grad_im[i] += 4.0 * AMP_WEIGHT * Ω_im[i] * diff_sq
        end
        
        # 2. Slew Penalty Gradients (Re)
        for i in 1:(length(Ω_re)-1)
            diff = Ω_re[i+1] - Ω_re[i]
            workspace_grad_re[i]   += -2.0 * SLEW_WEIGHT * diff
            workspace_grad_re[i+1] +=  2.0 * SLEW_WEIGHT * diff
        end
        
        # 3. Slew Penalty Gradients (Im)
        for i in 1:(length(Ω_im)-1)
            diff = Ω_im[i+1] - Ω_im[i]
            workspace_grad_im[i]   += -2.0 * SLEW_WEIGHT * diff
            workspace_grad_im[i+1] +=  2.0 * SLEW_WEIGHT * diff
        end
        
        return [workspace_grad_re, workspace_grad_im]
    end

    # --- RUN OPTIMIZATION ---
    result = GRAPE.optimize(
        trajectories, tlist;
        J_T = custom_J_T,               
        chi = custom_chi,               
        J_a = J_a_penalty,              
        grad_J_a = custom_grad_J_a,      
        prop_method = QuantumPropagators.Cheby,
        maxiters = 150,                 
        show_trace = true 
    )
    
    Ω_re_opt = result.optimized_controls[1]
    Ω_im_opt = result.optimized_controls[2] 
    println("Optimization finished. Final Cost: ", result.J_T)

    phase_grape = atan.(Ω_im_opt[1:Nt-1], Ω_re_opt[1:Nt-1])
    phase_const = zeros(Nt - 1)
    t_pulse = tlist[1:Nt-1] 

    # --- 4. Time Evolution (Simulation) ---
    println("Simulating plotting state...")
    a_op = ops.Iq ⊗ ops.a
    P1_op = ops.P1 ⊗ ops.I_m 
    
    ψq_plot = sin(π/8) * ops.ket0 + 1im * cos(π/8) * ops.ket1
    ψ_start = ψq_plot ⊗ fock(0, Nm)
    
    ψ_const = copy(ψ_start)
    ψ_grape = copy(ψ_start)
    
    x_const, p_const, n_const, q1_const = Float64[], Float64[], Float64[], Float64[]
    x_grape, p_grape, n_grape, q1_grape = Float64[], Float64[], Float64[], Float64[]
    
    for k in 1:Nt-1
        push!(x_const, real(dot(ψ_const, a_op * ψ_const)))
        push!(p_const, imag(dot(ψ_const, a_op * ψ_const)))
        push!(n_const, real(dot(ψ_const, n_op * ψ_const)))
        push!(q1_const, real(dot(ψ_const, P1_op * ψ_const)))
        
        push!(x_grape, real(dot(ψ_grape, a_op * ψ_grape)))
        push!(p_grape, imag(dot(ψ_grape, a_op * ψ_grape)))
        push!(n_grape, real(dot(ψ_grape, n_op * ψ_grape)))
        push!(q1_grape, real(dot(ψ_grape, P1_op * ψ_grape)))
        
        H_const = H0 + Ω_const * HΩ_re
        ψ_const = exp(-1im * dt * H_const) * ψ_const
        
        H_grape = H0 + Ω_re_opt[k] * HΩ_re + Ω_im_opt[k] * HΩ_im
        ψ_grape = exp(-1im * dt * H_grape) * ψ_grape
    end
    
    # Final step measurement 
    push!(x_const, real(dot(ψ_const, a_op * ψ_const)))
    push!(p_const, imag(dot(ψ_const, a_op * ψ_const)))
    push!(n_const, real(dot(ψ_const, n_op * ψ_const)))
    push!(q1_const, real(dot(ψ_const, P1_op * ψ_const)))
    
    push!(x_grape, real(dot(ψ_grape, a_op * ψ_grape)))
    push!(p_grape, imag(dot(ψ_grape, a_op * ψ_grape)))
    push!(n_grape, real(dot(ψ_grape, n_op * ψ_grape)))
    push!(q1_grape, real(dot(ψ_grape, P1_op * ψ_grape)))

    t_sim = tlist 

    # --- 5. Plotting (4-Panel Dashboard) ---
    c = :purple
    
    p_phase = plot(title="Phase Space Trajectory", xlabel="⟨x⟩ / (2x₀)", ylabel="⟨p⟩ / (2p₀)", aspect_ratio=:equal, grid=true)
    plot!(p_phase, x_const, p_const, label="Const", lw=1.5, ls=:dash, color=:gray)
    plot!(p_phase, x_grape, p_grape, label="GRAPE", lw=2.5, color=c)
    scatter!(p_phase, [x_const[1], x_const[end]], [p_const[1], p_const[end]], color=:gray, marker=:circle, label="")
    scatter!(p_phase, [x_grape[1], x_grape[end]], [p_grape[1], p_grape[end]], color=c, marker=:circle, label="")

    p_phonon = plot(title="Average Phonons ⟨n⟩", xlabel="Time (t)", ylabel="⟨n⟩", grid=true)
    plot!(p_phonon, t_sim, n_const, label="Const", lw=1.5, ls=:dash, color=:gray)
    plot!(p_phonon, t_sim, n_grape, label="GRAPE", lw=2.5, color=c)

    p_pulse = plot(title="Laser Phase φ(t)", xlabel="Time (t)", ylabel="Phase (rad)", grid=true)
    plot!(p_pulse, t_pulse, phase_const, label="Const", lw=1.5, ls=:dash, color=:gray)
    plot!(p_pulse, t_pulse, phase_grape, label="GRAPE", lw=2.5, color=c)

    p_qubit = plot(title="Qubit Population P(|1⟩)", xlabel="Time (t)", ylabel="Probability", grid=true)
    plot!(p_qubit, t_sim, q1_const, label="Const", lw=1.5, ls=:dash, color=:gray)
    plot!(p_qubit, t_sim, q1_grape, label="GRAPE", lw=2.5, color=c)

    fig_combined = plot(p_phase, p_phonon, p_pulse, p_qubit, layout=(2,2), size=(1200, 800), margin=6Plots.mm)
    display(fig_combined)
    
    out_dir = plotsdir("PhaseSpaceStudy", "eta_$(η)")
    mkpath(out_dir) 
    out_file = joinpath(out_dir, "grape_dashboard_omega_$(ω).png")
    savefig(fig_combined, out_file)
    println("\nPlot successfully saved to: ", out_file)
end

compare_phase_space()