using NetworkTopology
using LinearAlgebra
using Plots
using JSON
using JLD
using NPZ
using Statistics
using LaTeXStrings
using LsqFit
using NaNStatistics


"""
Evaluate reconstruction performance of pruning
    args: arguments containing default values
    runs: number of parallel pruned runs
"""

function eval_pruning_measure(args, runs)
    Loss = []
    Dstsp = []
    Dhell = []
    MSE = []

    for i in 1:runs
        metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], BPTT.Utilities.format_run_ID(i), "prune_metrics.jld"))["prune_metrics"]
        metrics = Matrix(mapreduce(permutedims,vcat, metrics)')
        metrics[1,isnan.(metrics[3,:])] .= NaN #Catch NaNs
        metrics[2,isnan.(metrics[3,:])] .= NaN
        metrics[4,isnan.(metrics[3,:])] .= NaN

        push!(Loss, metrics[1,1:end])
        push!(Dstsp, metrics[2,1:end])
        push!(Dhell, metrics[3,1:end])
        push!(MSE, metrics[4,1:end])

    end

    return Loss, Dstsp, Dhell, MSE
end


load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();
runs = 1
Loss, Dstsp, Dhell, MSE = eval_pruning_measure(args, runs)
Dstsp=sort(mapreduce(permutedims,vcat,Dstsp),dims=1)[1:end,:]
Dhell=sort(mapreduce(permutedims,vcat,Dhell),dims=1)[1:end,:]
MSE=sort(mapreduce(permutedims,vcat,MSE),dims=1)[1:end,:]


params = 100 .* 0.8 .^ LinRange(0,args["prune_steps"]-1,args["prune_steps"])
start = 1
ende = args["prune_steps"]

p1 = plot(nanmean(Dstsp,dims=1)[start:ende],yerror=nansem(Dstsp,dims=1)[start:ende],
    xlabel="% weights remaining",ylabel=L"$D_{stsp}$",
    lw=4,marker=:dot, markersize=10,msw=3,markerstrokecolor=palette(:default)[1],color=palette(:default)[1],
    tickfontsize=18, labelfontsize=20,legendfontsize=18,size=(1000,550),margin=10Plots.mm,
    xticks=([1:3.0:length(params);],string.(round.(params[1:3:end],digits=0))))
p2 = plot(nanmean(Dhell,dims=1)[start:ende],yerror=nansem(Dhell,dims=1)[start:ende],
    label="Geometry-based",xlabel="% weights remaining",ylabel=L"$D_{H}$",
    lw=4,marker=:dot, markersize=10,msw=3,markerstrokecolor=palette(:default)[1],color=palette(:default)[1],
    ylim=(-0.005,Inf),tickfontsize=18, labelfontsize=20,legendfontsize=18,size=(1000,550),margin=10Plots.mm,
    xticks=([1:3.0:length(params);],string.(round.(params[1:3:end],digits=0))))

title = plot(title = args["name"], grid = false, showaxis = false, bottom_margin = -80Plots.px,titlefontsize=30)
plot(title,p1,p2,size=(2000,450),margin=15Plots.mm,layout = @layout([A{0.01h}; [B C]]))
savefig("plots/pruning_results.pdf")
