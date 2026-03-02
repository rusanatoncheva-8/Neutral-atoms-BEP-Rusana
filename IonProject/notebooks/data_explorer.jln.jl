# %% [markdown]
# # Ion Gate Optimization: Data Explorer
# This notebook uses DrWatson to automatically catalog all `.jld2` simulation files 
# in the `data/sims/` directory.

# %%
using DrWatson
@quickactivate "IonProject" # Ensures you are in the correct environment

using DataFrames
using Plots
using JLD2

# Include your custom plotting module
include(srcdir("Plotting.jl"))

# %% [markdown]
# ## 1. Load and Inspect All Data
# `collect_results` scans your data folder, extracts the parameters DrWatson saved, 
# and builds a DataFrame. This acts as a searchable database of all your runs.

# %%
# Load all results into a dataframe
df = collect_results(datadir("sims"))

println("Total experiments found: ", nrow(df))

# Display a clean summary table of the key parameters
if nrow(df) > 0
    display(df[:, ["mode", "T", "Nm", "nbar", "seed"]])
else
    println("No data found yet. Run the optimization script first!")
end

# %% [markdown]
# ## 2. Filter and Compare Runs
# Let's say you want to compare how fast the optimizer converged for different control modes 
# (e.g., `full` vs `omega_only`) while keeping the motion cutoff and thermal occupation the same.

# %%
# Filter for specific experiment conditions
target_Nm = 6
target_nbar = 0.2

df_filtered = filter(row -> row[:Nm] == target_Nm && row[:nbar] == target_nbar, df)

# Plot the infidelity (J_T) across iterations for all matching runs
p_compare = plot(
    title="Convergence Comparison (Nm=$target_Nm, nbar=$target_nbar)", 
    xlabel="Iteration", 
    ylabel="Infidelity (J_T)", 
    yscale=:log10,
    legend=:topright,
    grid=true,
    lw=2
)

for row in eachrow(df_filtered)
    # Extract the GRAPE history from the payload
    history = row[:payload]["history"]
    
    if !isempty(history)
        iters = [h.iter for h in history]
        Js = [h.J for h in history]
        
        # Create a descriptive label for the legend
        run_label = "$(row[:mode]) (T=$(row[:T])ns)"
        
        plot!(p_compare, iters, Js, label=run_label)
    end
end

display(p_compare)

# Optional: Save this comparison plot
# savefig(p_compare, plotsdir("convergence_comparison.png"))

# %% [markdown]
# ## 3. Deep Dive: Analyze a Specific Run
# Now, let's grab the best performing run from our filtered list and generate 
# all the detailed diagnostic plots (Bloch spheres, populations, gate success) 
# using the `Plotting.jl` module.

# %%
# Find the row with the lowest final infidelity in our filtered set
best_run_row = sort(df_filtered, :J_T)[1, :]

println("Visualizing best run:")
println("Mode: ", best_run_row[:mode])
println("Time: ", best_run_row[:T], " ns")

# Extract the payload dictionary
payload = best_run_row[:payload]

# Generate all figures using your Plotting module
# Set viz_qubit = 0 to see the |0> ⊗ |n=0> trajectory, 1 for |1> ⊗ |n=0>, etc.
figs = Plotting.plot_run(payload; viz_qubit=0)

# %% [markdown]
# ### Control Pulses
# %%
display(figs.fig_opt_ctrls)

# %% [markdown]
# ### Qubit Populations & Motional Expectation
# %%
display(figs.fig_pop)
display(figs.fig_n)

# %% [markdown]
# ### Gate Success & Motional Distribution
# %%
display(figs.fig_ptarget)
display(figs.fig_Pmn)

# %% [markdown]
# ### Coherence Diagnostics
# %%
display(figs.fig_bloch)
display(figs.fig_blen)