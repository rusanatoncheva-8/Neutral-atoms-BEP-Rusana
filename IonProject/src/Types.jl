module Types

export OptimState

"""
Snapshot of the optimizer state at one iteration (IQ-only).

We store interval controls (typically length Nt-1).
"""
struct OptimState
    iter::Int
    runtime_s::Float64
    Ω_re::Vector{Float64}
    Ω_im::Vector{Float64}
    J::Float64
    grad_norm::Float64
end

end # module
