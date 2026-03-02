module IO

using Dates

export timestamp_str, append_details!

timestamp_str() = Dates.format(Dates.now(), "ddmm-HHMM")

function append_details!(details_path::AbstractString, header::AbstractString, lines::Vector{String})
    open(details_path, "a") do io
        println(io, header)
        for ln in lines
            println(io, "    ", ln)
        end
    end
    return nothing
end

end # module
