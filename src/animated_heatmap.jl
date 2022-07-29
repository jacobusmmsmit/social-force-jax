using Plots
using DelimitedFiles

twod_data = readdlm("data/reshaped_density.csv", ',', Float32)
frames = 30
data = reshape(twod_data, frames, 20, 12)
plot_data = permutedims(data, (1, 3, 2))
cmax = maximum(plot_data) # Ensure the colourbar has the same limits
animated_heatmap = @animate for i in 1:frames
    heatmap(plot_data[i, :, :], clims=(0, cmax))
end
gif(animated_heatmap, "plots/animated_heatmap_shape=1.5.gif", fps = 30)