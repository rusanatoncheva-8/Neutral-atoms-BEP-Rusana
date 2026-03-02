module IonModel

using LinearAlgebra
import LinearAlgebra: kron
using QuantumPropagators
using QuantumPropagators: hamiltonian

export IonOps, build_ion_ops, ion_hamiltonian_IQ, ion_hamiltonian_ΩΔ, Rx, fock, ⊗

const 𝕚 = 1im
⊗(A,B) = kron(A,B)

function ladder_ops(N::Int)
    a = zeros(ComplexF64, N, N)
    for n in 2:N
        a[n-1, n] = sqrt(n-1)
    end
    return a, a'
end

function Rx(θ)
    σx = ComplexF64[0 1; 1 0]
    exp(-𝕚 * θ/2 * σx)
end

function fock(n, N)
    v = zeros(ComplexF64, N)
    v[n+1] = 1
    return v
end

struct IonOps
    Nm::Int
    ω::Float64
    η::Float64
    a::Matrix{ComplexF64}
    adag::Matrix{ComplexF64}
    I_m::Matrix{ComplexF64}
    ket0::Vector{ComplexF64}
    ket1::Vector{ComplexF64}
    P0::Matrix{ComplexF64}
    P1::Matrix{ComplexF64}
    σ01::Matrix{ComplexF64}
    σ10::Matrix{ComplexF64}
    Iq::Matrix{ComplexF64}
    Eplus::Matrix{ComplexF64}
    Eminus::Matrix{ComplexF64}
end

function build_ion_ops(; Nm::Int, ω::Real, η::Real)
    a, adag = ladder_ops(Nm)
    I_m = Matrix{ComplexF64}(I, Nm, Nm)

    ket0 = ComplexF64[1,0]
    ket1 = ComplexF64[0,1]
    P0 = ket0 * ket0'
    P1 = ket1 * ket1'
    σ01 = ket0 * ket1'
    σ10 = ket1 * ket0'
    Iq  = Matrix{ComplexF64}(I, 2, 2)

    X = a + adag
    exp_iηX(η_) = exp(𝕚 * η_ * X)
    Eplus  = exp_iηX(η)
    Eminus = exp_iηX(-η)

    return IonOps(Nm, float(ω), float(η), a, adag, I_m,
                  ket0, ket1, P0, P1, σ01, σ10, Iq, Eplus, Eminus)
end

"""
IQ-only Hamiltonian generator:
  H(t) = H0 + Ω_re(t) * HΩ_re + Ω_im(t) * HΩ_im

Optional: fixed detuning Δ0 inside drift (default 0.0).
"""
function ion_hamiltonian_IQ(ops::IonOps; Ω_re, Ω_im, Δ0::Real=0.0)
    # Drift: motional harmonic oscillator + optional fixed detuning on |1>
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m

    HΔ = ops.P1 ⊗ ops.I_m
    H0 = H0 + float(Δ0) * HΔ

    # Controls
    HΩ_re = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)
    HΩ_im = 𝕚 * ((ops.σ10 ⊗ ops.Eplus) - (ops.σ01 ⊗ ops.Eminus))

    return hamiltonian(H0, (HΩ_re, Ω_re), (HΩ_im, Ω_im))
end

function ion_hamiltonian_ΩΔ(ops::IonOps; Ω, Δ, Δ0::Real=0.0)
    # Drift: motional harmonic oscillator + optional fixed detuning on |1>
    H0_m = ops.ω * (ops.adag * ops.a)
    H0   = ops.Iq ⊗ H0_m

    HΔ = ops.P1 ⊗ ops.I_m
    H0 = H0 + float(Δ0) * HΔ

    # Single (real) drive quadrature
    HΩ = (ops.σ10 ⊗ ops.Eplus) + (ops.σ01 ⊗ ops.Eminus)

    # Controls: Ω(t) and Δ(t)
    return hamiltonian(H0, (HΩ, Ω), (HΔ, Δ))
end

end # module
