#!/usr/bin/env julia
using DrWatson
@quickactivate "IonProject"

using ArgParse
using LinearAlgebra
using Random
using Printf
using JLD2
using Logging
using GRAPE
using QuantumControl
using QuantumControl.Functionals: J_T_sm
using QuantumPropagators
using QuantumPropagators.Amplitudes: ShapedAmplitude
using QuantumPropagators.Shapes: flattop
using QuantumPropagators: Cheby
using Optim
using LineSearches

# Ensure we load modules from the src/ directory
include(srcdir("Types.jl"))
include(srcdir("IonModel.jl"))
include(srcdir("IO.jl"))
include(srcdir("Utils.jl"))

using .Types: OptimState
using .IonModel: build_ion_ops, ion_hamiltonian_ΩΔ, Rx, fock, ⊗
using .IO: timestamp_str, append_details!
using .Utils: safe_grad_norm, try_get_controls

const GHz = 1.0
const MHz = 0.001GHz
const ns  = 1.0
const WALL_WEIGHT = 10000.0
const DIFF_WEIGHT = 0.01

# pass variables into your Julia script directly from your computer's terminal
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--control_mode"
            help = "Choose: 'full' (Ω and Δ), 'omega_only', or 'constant'"
            arg_type = String
            default = "full"
        "--T"
            help = "Total time in ns"
            arg_type = Float64
            default = 360.0
        "--Nm"
            help = "Motional cutoff"
            arg_type = Int
            default = 6
        "--nbar"
            help = "Mean thermal occupation"
            arg_type = Float64
            default = 0.2
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 1234
    end
    return parse_args(s)
end

# thermal distribution
function thermal_probs(Nm::Int; nbar::Real)
    q = nbar / (1 + nbar)                 
    p = [(1 - q) * q^n for n in 0:Nm-1]    
    s = sum(p)
    return p ./ s
end

function main()
    args = parse_commandline()
    
    # Unpack arguments
    mode = args["control_mode"]
    T_ns = args["T"]
    T = T_ns * ns
    Nm = args["Nm"]
    nbar = args["nbar"]
    seed = args["seed"]

    # DrWatson parameter dictionary for smart saving
    params = Dict(
        "mode" => mode,
        "T" => T_ns,
        "Nm" => Nm,
        "nbar" => nbar,
        "seed" => seed
    )

    # Calculate Nt and tlist based on mode
    if mode == "constant"
        dt = T_ns * ns
        tlist = collect(0:dt:T)
        Nt = 1
    else
        dt = (T/400) * ns
        tlist = collect(0:dt:T)
        Nt = length(tlist)
    end
    
    nseg = Nt > 1 ? length(tlist) - 1 : 1

    rectshape(t) = 1.0
    ω  = 2π * 2.5 * MHz
    η  = 0.1
    θ = π
    Ωmax = 500 * 2π * MHz
    Ω0 = 35 * 2π * MHz
    Δmax = 200 * 2π * MHz
    
    J_star = 1e-6
    grad_star = 1e-6
    maxiters = 2000

    rng = MersenneTwister(seed)
    p_th = thermal_probs(Nm; nbar=nbar)
    
    function haar_state(d::Int; rng=Random.default_rng())
        v = randn(rng, ComplexF64, d)
        return v / norm(v)
    end

    ops = build_ion_ops(; Nm=Nm, ω=ω, η=η)

    ψq1 = haar_state(2; rng=rng)
    ψq2 = haar_state(2; rng=rng)

    qubit_inputs = [ops.ket0, ops.ket1, ψq1, ψq2]
    Xq = Rx(θ)
    qubit_targets = [Xq * q for q in qubit_inputs]
    P_targets = [(qf*qf') ⊗ ops.I_m for qf in qubit_targets]

    # Full hamiltonian
    if mode == "full"
        ϵ_Ω_guess(t) = Ω0
        ϵ_Δ_guess(t) = 0.0
        Ω = ShapedAmplitude(ϵ_Ω_guess; shape=rectshape)
        Δ = ShapedAmplitude(ϵ_Δ_guess; shape=rectshape)
        Hgen = ion_hamiltonian_ΩΔ(ops; Ω=Ω, Δ=Δ)
    else # omega_only or constant
        ϵ_Ω_guess(t) = Ω0
        Ω = ShapedAmplitude(ϵ_Ω_guess; shape=rectshape)
        H0_m = ops.ω * (ops.adag * ops.a)
        H0   = ops.Iq ⊗ H0_m
        HΩ   = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
        Hgen = QuantumPropagators.hamiltonian(H0, (HΩ, Ω))
    end

    trajectories = QuantumControl.Trajectory[]
    weights = Float64[]

    # Optimisation for 4*Nm states; for every motional level, the code initialises 4 distinct qubit states, each paired with that specific Fock state
    # the optimiser put different weights on the different states depending on the average motional number nbar
    for n in 0:Nm-1
        w = p_th[n+1]
        ψm = fock(n, Nm)

        push!(trajectories, QuantumControl.Trajectory(ops.ket0 ⊗ ψm, Hgen; P_target=P_targets[1]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(ops.ket1 ⊗ ψm, Hgen; P_target=P_targets[2]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(ψq1 ⊗ ψm, Hgen; P_target=P_targets[3]))
        push!(weights, w)
        push!(trajectories, QuantumControl.Trajectory(ψq2 ⊗ ψm, Hgen; P_target=P_targets[4]))
        push!(weights, w)
    end

    println("Optimization initialized in '$mode' mode with $(length(trajectories)) trajectories.")

    # The fidelity related functions
    function J_T_qubitproj(Ψ, trajectories)
        s = 0.0
        wsum = 0.0
        for k in eachindex(Ψ)
            w = weights[k]
            #  For each trajectory k, it calculates the overlap between the actual state at the end of the gate
            # Ψ[k] and the target state projector (P_target); considers the motional state weight
            s += w * real(dot(Ψ[k], trajectories[k].P_target * Ψ[k]))
            wsum += w
        end
        #  averages the success accroas all trajectories and substracts it from on (giving infidelity)
        return 1 - s / wsum
    end
 
    # The derivative of the cost function with respect to the quantum state
    function chi_qubitproj(Ψ, trajectories)
        χ = Vector{typeof(Ψ[1])}(undef, length(Ψ))
        wsum = sum(weights)
        for k in eachindex(Ψ)
            # scales the correction signal vy the thermal weights, so the optimiser prioritises fixing errors that matter most to the fidelity
            χ[k] = (weights[k] / wsum) * (trajectories[k].P_target * Ψ[k])
        end
        return χ
    end

    # smoothness penalty
    # it looks in the quadratic difference between the velocities at two neighboring locations; if the pulse jumps significantly - cost is increased
    function J_slew(pulsevals, tlist)
        nseg <= 1 && return 0.0
        penalty = 0.0
        for k in 1:nseg-1
            penalty += (pulsevals[k+1] - pulsevals[k])^2
        end
        if mode == "full" # to penelise both omega and detuning
            for k in 1:nseg-1
                idx = nseg + k
                penalty += (pulsevals[idx+1] - pulsevals[idx])^2
            end
        end
        return DIFF_WEIGHT * penalty
    end

    # the derivative of the penalty
    function grad_J_slew(pulsevals, tlist)
        grad = zeros(length(pulsevals))
        nseg <= 1 && return grad
        
        function compute_smooth_grad!(g, vals, offset)
            # for the boundary ends
            g[offset + 1] += 2 * (vals[1] - vals[2])
            for k in 2:nseg-1
                #gradient approximation
                g[offset + k] += 2 * (2 * vals[k] - vals[k-1] - vals[k+1])
            end
            g[offset + nseg] += 2 * (vals[nseg] - vals[nseg-1])
        end
        
        compute_smooth_grad!(grad, pulsevals[1:nseg], 0)
        if mode == "full"
            compute_smooth_grad!(grad, pulsevals[nseg+1:2nseg], nseg)
        end
        return DIFF_WEIGHT * grad
    end

    # manual boundary implementation
    function J_boundary(pulsevals, tlist)
        penalty = 0.0
        for i in 1:nseg
            v = pulsevals[i]
            if v < 0.0
                # quartic barrier
                penalty += v^4
            elseif v > Ωmax
                penalty += (v - Ωmax)^4
            end
        end
        if mode == "full"
            for i in 1:nseg
                idx = nseg + i
                v = pulsevals[idx]
                if v < -Δmax
                    penalty += (v - (-Δmax))^4
                elseif v > Δmax
                    penalty += (v - Δmax)^4
                end
            end
        end
        return WALL_WEIGHT * penalty
    end

    function grad_J_boundary(pulsevals, tlist)
        grad = zeros(length(pulsevals))
        for i in 1:nseg
            v = pulsevals[i]
            if v < 0.0
                # gradient of the quartic barrier
                grad[i] = 4 * v^3
            elseif v > Ωmax
                grad[i] = 4 * (v - Ωmax)^3
            end
        end
        if mode == "full"
            for i in 1:nseg
                idx = nseg + i
                v = pulsevals[idx]
                if v < -Δmax
                    grad[idx] = 4 * (v - (-Δmax))^3
                elseif v > Δmax
                    grad[idx] = 4 * (v - Δmax)^3
                end
            end
        end
        return WALL_WEIGHT * grad
    end

    function J_total_penalty(pulsevals, tlist)
        return J_boundary(pulsevals, tlist) + J_slew(pulsevals, tlist)
    end

    function grad_J_total_penalty(pulsevals, tlist)
        return grad_J_boundary(pulsevals, tlist) + grad_J_slew(pulsevals, tlist)
    end

    # optimisatioon setup
    history = NamedTuple[]
    t0 = time()

    function log_callback(wrk, iter)
        res = wrk.result
        runtime = time() - t0
        gnorm = norm(wrk.grad_J_T)

        pv = copy(wrk.pulsevals)
        Ωvals = pv[1:nseg]
        
        if mode == "full"
            Δvals = pv[nseg+1:2nseg]
            push!(history, (iter = iter, runtime_s = runtime, Ω = Ωvals, Δ = Δvals, J = res.J_T, grad_norm = gnorm))
        else
            push!(history, (iter = iter, runtime_s = runtime, Ω = Ωvals, J = res.J_T, grad_norm = gnorm))
        end

        Printf.@printf("iter=%4d  J=%.3e  |∇J|=%.3e  runtime=%.1fs\n", iter, res.J_T, gnorm, runtime)
        return (res.J_T, gnorm)
    end

    function check_convergence(res)
        if length(res.records) >= maxiters
            return "Reached maximum iterations ($maxiters)"
        end
        if !isempty(res.records)
            J_T_last, gT_norm_last = res.records[end]
            if (J_T_last < J_star) || (gT_norm_last < grad_star)
                return "J_T < 1e-6 OR ||∇J_T|| < 1e-6"
            end
        end
        return false
    end

    result = nothing
    err_str = nothing
    bt_str = nothing

    try
        result = GRAPE.optimize(
            trajectories, tlist;
            J_T = J_T_qubitproj,
            J_a = J_total_penalty,
            grad_J_a = grad_J_total_penalty,
            chi = chi_qubitproj,
            prop_method = Cheby,
            maxiters = maxiters,
            show_trace = true,
            print_iters = false,
            callback = log_callback,
            check_convergence = check_convergence,
            pulse_options = Dict(),  
            optimizer = Optim.LBFGS(linesearch = LineSearches.BackTracking(order=3)),
        )

        println("\nOptimization Return Status:")
        println(result)
    catch err
        err_str = sprint(showerror, err)
        bt_str  = sprint(show, catch_backtrace())
        @warn "Optimization crashed" err_str
        println(bt_str)
    finally
        total_runtime = time() - t0

        println("\n--- OPTIMIZATION FINISHED ---")
        if isempty(history)
            println("WARNING: History is empty! No iterations were recorded.")
        else
            println("Success: History contains $(length(history)) iterations.")
        end

        # --- DRWATSON AUTOMATIC SAVING ---
        fname = savename(params, "jld2")
        outpath = datadir("sims", fname)
        mkpath(dirname(outpath)) # Ensure the directory exists

        payload = Dict(
            "params" => params,
            "T" => T, "dt" => dt, "Nt" => Nt, "Nm" => Nm,
            "ω" => ω, "η" => η, "seed" => seed, "θ" => θ,
            "nbar" => nbar, "thermal_probs" => p_th,
            "K" => length(trajectories),
            "ψq1" => ψq1, "ψq2" => ψq2,
            "Ω_guess" => fill(Ω0, nseg),
            "history" => history, "result" => result,
            "error" => err_str, "backtrace" => bt_str
        )
        
        if mode == "full"
            payload["Δmax"] = Δmax
            payload["Δ_guess"] = fill(0.0, nseg)
        end

        wsave(outpath, Dict("payload" => payload))
        println("\nSaved results to: ", outpath)
    end
end

main()