module Utils

using LinearAlgebra
import LinearAlgebra: norm

export safe_grad_norm, try_get_controls

"""
Try to compute the gradient norm from the GRAPE workspace.
Returns NaN if the field isn't available in your GRAPE version.
"""
function safe_grad_norm(wrk)
    for name in (:grad_J_T, :gradJT, :grad)
        if hasproperty(wrk, name)
            g = getproperty(wrk, name)
            return g === nothing ? NaN : norm(g)
        end
    end
    return NaN
end

"""
Try to extract current control arrays from the workspace/result in a tolerant way.
Returns `nothing` if not found.
"""
function try_get_controls(wrk)
    candidates = (:pulsevals, :pulse_values, :controls, :control_values, :optimized_controls)
    for name in candidates
        if hasproperty(wrk, name)
            return getproperty(wrk, name)
        end
    end
    if hasproperty(wrk, :result) && hasproperty(wrk.result, :optimized_controls)
        return wrk.result.optimized_controls
    end
    return nothing
end

end # module
