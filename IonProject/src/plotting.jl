module Plotting

using LinearAlgebra
import LinearAlgebra: kron
using Plots
using JLD2
using Random
using Statistics

# Units
const GHz = 1.0
const MHz = 0.001GHz
const ns  = 1.0
const 𝕚   = 1im

# Include IonModel directly to reuse existing functions instead of duplicating them
include("IonModel.jl")
using .IonModel: build_ion_ops, Rx, fock, ⊗, ladder_ops

# -----------------------------
# Hamiltonian pieces for evaluation
# -----------------------------
function build_ion_pieces_eval(; Nm::Int, ω, η, Δ0::Real=0.0)
    a, adag = ladder_ops(Nm)
    I_m = Matrix{ComplexF64}(I, Nm, Nm)

    ket0 = ComplexF64[1,0]; ket1 = ComplexF64[0,1]
    P0 = ket0*ket0';  P1 = ket1*ket1'
    σ01 = ket0*ket1'; σ10 = ket1*ket0'
    Iq  = Matrix{ComplexF64}(I, 2, 2)

    X = a + adag
    exp_iηX(η_) = exp(𝕚 * η_ * X)
    Eplus, Eminus = exp_iηX(η), exp_iηX(-η)

    H0_m = ω * (adag * a)
    H0   = Iq ⊗ H0_m
    HΔ = P1 ⊗ I_m
    H0 = H0 + float(Δ0) * HΔ
    HΩ = (σ10 ⊗ Eplus) + (σ01 ⊗ Eminus)

    return (; a, adag, I_m, ket0, ket1, P0, P1, Iq, H0, HΩ, HΔ)
end

function evolve_trajectory(ψ0, Ω_vals, Δ_vals, tlist; H0, HΩ, HΔ)
    states = Vector{Vector{ComplexF64}}(undef, length(tlist))
    states[1] = copy(ψ0)

    for k in 1:length(tlist)-1
        Δt = tlist[k+1] - tlist[k]
        
        # Handle cases where Δ_vals might be empty (omega_only mode)
        Hk = H0 + Ω_vals[k]*HΩ
        if !isempty(Δ_vals)
            Hk += Δ_vals[k]*HΔ
        end
        
        states[k+1] = exp(-𝕚 * Δt * Hk) * states[k]
    end
    return states
end

expval(A, ψ) = real(dot(ψ, A*ψ))

function motional_probs(traj::Vector{Vector{ComplexF64}}, Nm::Int)
    out = Matrix{Float64}(undef, Nm, length(traj))
    for (k, ψ) in enumerate(traj)
        Ψ = reshape(ψ, Nm, 2) # (Nm, qubit=2)
        out[:, k] = vec(sum(abs2, Ψ; dims=2))
    end
    return out
end

p_target_over_time(traj_states, P) = [real(dot(ψ, P * ψ)) for ψ in traj_states]

function partial_trace_qubit(ψ::Vector{ComplexF64}, Nm::Int)
    M = reshape(ψ, Nm, 2)      
    ρ = M' * M                 
    ρ = (ρ + ρ')/2             
    trρ = real(tr(ρ))
    if trρ > 0
        ρ ./= trρ
    end
    return Hermitian(ρ)
end

function bloch_components(ρ::AbstractMatrix{ComplexF64})
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -1im; 1im 0]
    σz = ComplexF64[1 0; 0 -1]
    bx = real(tr(ρ * σx))
    by = real(tr(ρ * σy))
    bz = real(tr(ρ * σz))
    return bx, by, bz
end

function bloch_over_time(traj::Vector{Vector{ComplexF64}}, Nm::Int)
    bx = Float64[]; by = Float64[]; bz = Float64[]; bnorm = Float64[]
    for ψ in traj
        ρq = partial_trace_qubit(ψ, Nm)
        x, y, z = bloch_components(ρq)
        push!(bx, x); push!(by, y); push!(bz, z)
        push!(bnorm, sqrt(x^2 + y^2 + z^2))
    end
    return bx, by, bz, bnorm
end

function plot_bloch_components(t_ns, traj_states, Nm; title_str="Reduced-qubit Bloch components")
    bx, by, bz, _ = bloch_over_time(traj_states, Nm)
    p = plot(t_ns, bx; label="⟨σx⟩", lw=2, xlabel="time (ns)", ylabel="Bloch component", title=title_str)
    plot!(p, t_ns, by; label="⟨σy⟩", lw=2)
    plot!(p, t_ns, bz; label="⟨σz⟩", lw=2)
    plot!(p; ylims=(-1.05, 1.05))
    return p
end

function plot_bloch_length(t_ns, traj_states, Nm; title_str="Bloch length |b(t)|")
    _, _, _, bnorm = bloch_over_time(traj_states, Nm)
    p = plot(t_ns, bnorm; label="|b|", lw=2, xlabel="time (ns)", ylabel="|b|",
             title=title_str, ylims=(0, 1.05))
    return p
end

function plot_controls_stacked(t_mid, Ωvals, Δvals; title_str="", mode="full")
    pΩ = plot(t_mid ./ ns, Ωvals ./ (2π*MHz);
              xlabel=mode=="full" ? "" : "time (ns)", ylabel="Ω (2π·MHz)",
              label="Ω", title=title_str, lw=2)

    if mode == "full"
        pΔ = plot(t_mid ./ ns, Δvals ./ (2π*MHz);
                  xlabel="time (ns)", ylabel="Δ (2π·MHz)",
                  label="Δ", lw=2)
        return plot(pΩ, pΔ; layout=(2,1), link=:x)
    else
        return pΩ
    end
end

function plot_cost_and_grad_stacked(iters, Js, gs; title_str="Optimization trace")
    pJ = plot(iters, Js; yscale=:log10, xlabel="", ylabel="J_T", label="J_T", title=title_str, lw=2)
    pg = plot(iters, gs; yscale=:log10, xlabel="iteration", ylabel="||∇J||", label="||∇J||", lw=2)
    return plot(pJ, pg; layout=(2,1), link=:x)
end

function get_optimization_inputs(payload::Dict, pieces)
    Nm = payload["Nm"]
    θ  = get(payload, "θ", π)
    ψq1 = payload["ψq1"]
    ψq2 = payload["ψq2"]

    qubit_inputs = [pieces.ket0, pieces.ket1, ψq1, ψq2]
    Xq = Rx(θ)
    qubit_targets = [Xq * q for q in qubit_inputs]
    P_type = [(qf*qf') ⊗ pieces.I_m for qf in qubit_targets]

    initial_states = Vector{Vector{ComplexF64}}()
    P_targets      = Matrix{ComplexF64}[]
    labels         = String[]

    # Deterministic generation
    for n in 0:Nm-1
        ϕm = fock(n, Nm)
        for t in 1:4
             ψ0 = qubit_inputs[t] ⊗ ϕm
            push!(initial_states, ψ0)
            push!(P_targets, P_type[t])
            push!(labels, "$(t==1 ? "|0⟩" : t==2 ? "|1⟩" : t==3 ? "Haar1" : "Haar2") ⊗ |n=$n⟩")
        end
    end

    return initial_states, P_targets, labels
end

function gate_success_curves_banded(payload::Dict, pieces, tlist, t_ns, Ω_vals, Δ_vals)
    initial_states, P_targets, _ = get_optimization_inputs(payload, pieces)
    K = length(initial_states)

    curves = Vector{Vector{Float64}}(undef, K)
    for i in 1:K
        tr = evolve_trajectory(initial_states[i], Ω_vals, Δ_vals, tlist; H0=pieces.H0, HΩ=pieces.HΩ, HΔ=pieces.HΔ)
        curves[i] = p_target_over_time(tr, P_targets[i])
    end

    groups = Dict(
        1 => findall(i -> (i-1) % 4 == 0, 1:K),
        2 => findall(i -> (i-1) % 4 == 1, 1:K),
        3 => findall(i -> (i-1) % 4 == 2, 1:K),
        4 => findall(i -> (i-1) % 4 == 3, 1:K),
    )
    glabel = Dict(1=>"|0⟩", 2=>"|1⟩", 3=>"Haar1", 4=>"Haar2")

    fig = plot(title="Gate Success (mean/min/max over motional samples)",
               xlabel="time (ns)", ylabel="⟨P_target⟩", ylim=(0, 1.05), legend=:bottomright)

    for t in 1:4
        idxs = groups[t]
        isempty(idxs) && continue
        M = hcat(curves[idxs]...)
        mean_curve = vec(mean(M; dims=2))
        min_curve  = vec(minimum(M; dims=2))
        max_curve  = vec(maximum(M; dims=2))

        plot!(fig, t_ns, mean_curve, lw=3, label="$(glabel[t]) mean")
        plot!(fig, t_ns, min_curve,  lw=1, linestyle=:dash, label="$(glabel[t]) min")
        plot!(fig, t_ns, max_curve,  lw=1, linestyle=:dash, label="$(glabel[t]) max")
    end

    return fig
end

# Main entry point exposed to the notebook/scripts
function plot_run(payload::Dict; viz_qubit::Int=0, save_dir=nothing)
    history = payload["history"]
    isempty(history) && error("History is empty — cannot plot.")

    params = get(payload, "params", Dict("mode" => "full"))
    mode = get(params, "mode", "full")

    T   = payload["T"]
    dt  = payload["dt"]
    Nm  = payload["Nm"]
    ω   = payload["ω"]
    η   = payload["η"]

    tlist = collect(0:dt:T)
    t_mid = @.(tlist[1:end-1] + tlist[2:end]) / 2
    t_ns  = tlist ./ ns

    st = history[end]
    Ω_opt_int = st.Ω
    Δ_opt_int = mode == "full" ? st.Δ : Float64[]

    Ω_guess_int = get(payload, "Ω_guess", history[1].Ω)
    Δ_guess_int = mode == "full" ? get(payload, "Δ_guess", history[1].Δ) : Float64[]

    fig_guess_ctrls = plot_controls_stacked(t_mid, Ω_guess_int, Δ_guess_int; title_str="Initial guess", mode=mode)
    fig_opt_ctrls   = plot_controls_stacked(t_mid, Ω_opt_int, Δ_opt_int; title_str="Optimized", mode=mode)

    pieces = build_ion_pieces_eval(; Nm=Nm, ω=ω, η=η)

    initial_states, _, labels = get_optimization_inputs(payload, pieces)
    K = length(initial_states)
    idx_viz = clamp(viz_qubit + 1, 1, K)
    ψ0_viz = initial_states[idx_viz]
    lbl_viz = labels[idx_viz]

    traj = evolve_trajectory(ψ0_viz, Ω_opt_int, Δ_opt_int, tlist; H0=pieces.H0, HΩ=pieces.HΩ, HΔ=pieces.HΔ)

    P0_op = pieces.P0 ⊗ pieces.I_m
    P1_op = pieces.P1 ⊗ pieces.I_m
    n_op  = pieces.Iq ⊗ (pieces.adag * pieces.a)

    p0   = [expval(P0_op, ψ) for ψ in traj]
    p1   = [expval(P1_op, ψ) for ψ in traj]
    nexp = [expval(n_op,  ψ) for ψ in traj]
    Pmn  = motional_probs(traj, Nm)

    fig_pop = plot(t_ns, p0; label="P(|0⟩)", xlabel="time (ns)", ylabel="population", title="Qubit populations ($(lbl_viz))", lw=2)
    plot!(t_ns, p1; label="P(|1⟩)", legend=:right, lw=2)

    fig_n = plot(t_ns, nexp; label="⟨n⟩", xlabel="time (ns)", ylabel="⟨n⟩", title="Motional ⟨n⟩ ($(lbl_viz))", lw=2)

    fig_Pmn = heatmap(t_ns, 0:Nm-1, Pmn; xlabel="time (ns)", ylabel="n", title="Motional distribution p(n,t)", colorbar_title="prob")

    fig_ptarget = gate_success_curves_banded(payload, pieces, tlist, t_ns, Ω_opt_int, Δ_opt_int)

    fig_bloch = plot_bloch_components(t_ns, traj, Nm; title_str="Bloch components ($(lbl_viz))")
    fig_blen  = plot_bloch_length(t_ns, traj, Nm; title_str="Bloch length |b| ($(lbl_viz))")

    iters = [h.iter for h in history]
    Js    = [h.J for h in history]
    gs    = [h.grad_norm for h in history]
    fig_costgrad = plot_cost_and_grad_stacked(iters, Js, gs; title_str="Optimization trace")

    if save_dir !== nothing
        mkpath(save_dir)
        savefig(fig_guess_ctrls, joinpath(save_dir, "guess_controls.png"))
        savefig(fig_opt_ctrls,   joinpath(save_dir, "opt_controls.png"))
        savefig(fig_costgrad, joinpath(save_dir, "cost_grad.png"))
        savefig(fig_pop,      joinpath(save_dir, "populations.png"))
        savefig(fig_n,        joinpath(save_dir, "n_expectation.png"))
        savefig(fig_Pmn,      joinpath(save_dir, "motional_distribution.png"))
        savefig(fig_ptarget,  joinpath(save_dir, "gate_success_trajectories.png"))
        savefig(fig_bloch,    joinpath(save_dir, "bloch_components.png"))
        savefig(fig_blen,     joinpath(save_dir, "bloch_length.png"))
    end

    return (
        fig_guess_ctrls = fig_guess_ctrls,
        fig_opt_ctrls   = fig_opt_ctrls,
        fig_costgrad    = fig_costgrad,
        fig_pop         = fig_pop,
        fig_n           = fig_n,
        fig_Pmn         = fig_Pmn,
        fig_ptarget     = fig_ptarget,
        fig_bloch       = fig_bloch,
        fig_blen        = fig_blen,
    )
end

end # module