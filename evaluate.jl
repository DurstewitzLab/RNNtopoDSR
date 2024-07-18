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
using StatsPlots
using Graphs


"""
Evaluate single model
Calculate metrics and plot generated trajectory
"""

function evaluate_run(model, O, data, measure_settings::AbstractDict)

    # Initialize
    X = data
    T = size(X, 1)
    z₁ = Float32.(init_state(O, X[1, :]))

    # generate trajectory
    X_gen = @views generate(model, O, X[1, :], T)
    Z = @views generate(model, z₁, T)

    # metrics
    Dstsp = NetworkTopology.evaluate_Dstsp(X, X_gen, measure_settings["Dstsp"])
    PSE, _ = NetworkTopology.evaluate_PSE(X, X_gen, measure_settings["PSE"])
    PE = NetworkTopology.evaluate_PE(model, O, X, measure_settings["PE"])

    return Dict("Dstsp" => Dstsp, "PSE" => PSE, "PE" => PE), X_gen, Z
end


# Load arguments
load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();

# load metrics
metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "metrics.jld"))["metrics"] # load metrics
best_id = argmin(Matrix(mapreduce(permutedims,vcat, metrics)'),dims=2)[2][2] #Get best model 

# data and model
data = Float32.(npzread(joinpath(pwd(), "example_data", "lorenz63.npy")))
model, O = load_model(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "checkpoints", "model_"*string(best_id*args["scalar_saving_interval"])*".bson"))

# Evaluate
measure_settings = Dict("Dstsp" => 30, "PSE" => 20.0f0, "PE" => 20);
res, X_gen, Z = evaluate_run(model, O, data, measure_settings);
println("Measure results:\n",res)

#Plotting
lim = 2000
plot(data[:,1], data[:,2], data[:,3], label="ground truth",xlabel=L"x",ylabel=L"y",zlabel=L"z",lw=3,color=palette(:default)[1])
p1 = plot!(X_gen[:,1], X_gen[:,2], X_gen[:,3],xlabel=L"x",ylabel=L"y",zlabel=L"z", label="generated", lw=3,color=palette(:default)[2])
plot(data[1:lim,1],label="",xlabel=L"t",ylabel=L"x",lw=3)
p2 = plot!(X_gen[1:lim,1],label="",xlabel=L"t",ylabel=L"x",lw=3)
plot(data[1:lim,2],label="",xlabel=L"t",ylabel=L"y",lw=3)
p3 = plot!(X_gen[1:lim,2],label="",xlabel=L"t",ylabel=L"y",lw=3)
plot(data[1:lim,3],label="",xlabel=L"t",ylabel=L"z",lw=3)
p4 = plot!(X_gen[1:lim,3],label="",xlabel=L"t",ylabel=L"z",lw=3)
plot(p1,p2,p3,p4,size=(1500,700),margin = 5Plots.mm, tickfontsize=18, labelfontsize=20,legendfontsize=20,margins=8Plots.mm,layout = @layout [a{0.4w} [b ; c; d]])




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
        metrics = load(joinpath(pwd(), "Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "prune_metrics.jld"))["prune_metrics"] # load metrics
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
runs = 1 # number of parallel pruned model runs
Loss, Dstsp, Dhell, MSE = eval_pruning_measure(args, runs)
Dstsp=sort(mapreduce(permutedims,vcat,Dstsp),dims=1)[1:end,:]
Dhell=sort(mapreduce(permutedims,vcat,Dhell),dims=1)[1:end,:]
MSE=sort(mapreduce(permutedims,vcat,MSE),dims=1)[1:end,:]


params = 100 .* 0.8 .^ LinRange(0,args["prune_steps"]-1,args["prune_steps"]) # % weights remaining
start = 1
ende = args["prune_steps"]

# Plotting
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


"""
Calculate graph structural properties of pruned networks
"""

#Provide model arguments and iteration number of the pruned model during iterative pruning
iteration = string(12) # models with 92% weights removed (for standard setting)
load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))
args = load_defaults();


# Calculate cummulative degree of network
# range controls the histogram binning, runs is the amount of pruned models that should get evaluated
function cummulative_degree_eval(args, iteration, range = 0:0.02:1.0, runs=1)
    M = args["latent_dim"]
    hist_in = zeros(length(range)-1); hist_out = zeros(length(range)-1)

    for i in 1:runs
        model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "prune_model_"*iteration*".bson"))
        hist_in += histcounts(Graphs.indegree(DiGraph(model.W_mask))./(size(model.W_mask)[1]-1),range)
        hist_out += histcounts(Graphs.outdegree(DiGraph(model.W_mask))./(size(model.W_mask)[1]-1),range)
    end
    hist_in ./= runs
    hist_out ./= runs
    
    return reverse(cumsum(reverse(hist_in)))./M .+ 0.001, reverse(cumsum(reverse(hist_out)))./M .+ 0.001

end

hist_in, hist_out = cummulative_degree_eval(args,iteration)

# For comparison calculate the cummulative degree for other graph structures
runs = 100; M=50; K_WS=4; K_BA=2; range=0:0.02:1.0
hist_WS = zeros(length(range)-1); hist_BA = zeros(length(range)-1); hist_ER = zeros(length(range)-1)

for i in 1:runs
    G_M = NetworkTopology.watts_strogatz_graph(M,K_WS,0.1,3)
    hist_WS += histcounts(Graphs.outdegree(DiGraph(G_M))./(size(G_M)[1]-1),range) # Watts-Strogatz model
    G_M = NetworkTopology.erdos_renyi_graph(M,K_BA,0.1,3)
    hist_ER += histcounts(Graphs.outdegree(DiGraph(G_M))./(size(G_M)[1]-1),range) # Erdos-Renyi model
    G_M = NetworkTopology.barabasi_albert_graph(M,K_BA,1,1)
    hist_BA += histcounts(Graphs.indegree(DiGraph(G_M))./(size(G_M)[1]-1),range) # Barabasi-Albert model
end
hist_WS ./= runs
hist_WS = reverse(cumsum(reverse(hist_WS)))./M .+ 0.001
hist_ER ./= runs
hist_ER = reverse(cumsum(reverse(hist_ER)))./M .+ 0.001
hist_BA ./= runs
hist_BA = reverse(cumsum(reverse(hist_BA)))./M .+ 0.001

#Plotting
plot(range[2:end],hist_WS,lw=7,color=palette(:tab10)[4],yaxis=:log,ylim=(8*10^-3,1.2),xlim=(range[2]-0.02,0.4),label="Watts-Strogatz",ylabel=L"F(k')",xlabel=L"k'")
plot!(range[2:end],hist_ER,lw=7,color=palette(:tab10)[2],label="Erdős-Rényi",legend=:bottomleft)
plot!(range[2:end],hist_BA,lw=7,color=palette(:tab10)[5],label="Barabási-Albert",legend=:bottomleft)
plot!(range[2:end],hist_out,lw=7,color=palette(:tab10)[10],label="out-degree",legend=:bottomleft)
p1 = plot!(range[2:end],hist_in,lw=7,color=palette(:tab10)[1],label="in-degree",legend=:topright)


# Calculate degree dependent on readout and latent nodes.
# N is data dimension, range controls the histogram binning, runs is the amount of pruned models that should get evaluated
function hub_hist_eval(args, N, iteration, runs=1)

    c_in_obs = []; c_in_lat = [];c_out_obs = []; c_out_lat = [];

    for i in 1:runs
        model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "prune_model_"*iteration*".bson"))
        push!(c_in_obs,Graphs.indegree_centrality(DiGraph(model.W_mask))[1:N])
        push!(c_in_lat,Graphs.indegree_centrality(DiGraph(model.W_mask))[N+1:end])
        push!(c_out_obs,Graphs.outdegree_centrality(DiGraph(model.W_mask))[1:N])
        push!(c_out_lat,Graphs.outdegree_centrality(DiGraph(model.W_mask))[N+1:end])
    end
    
    return reduce(vcat,c_in_obs),reduce(vcat,c_in_lat),reduce(vcat,c_out_obs),reduce(vcat,c_out_lat)
end

c_in_obs,c_in_lat,c_out_obs,c_out_lat = hub_hist_eval(args,3,iteration)
c_obs = (c_in_obs + c_out_obs)/2
c_lat = (c_in_lat + c_out_lat)/2

# Plotting
histogram(c_obs,bins=0:0.05:0.7,normalize=:pdf,xlabel=L"k'",ylabel=L"P(k')",label="Readout nodes",color=palette(:tab10)[1],yticks=([1:1:7;],string.(round.(1*0.05:1.0*0.05:7*0.05,digits=2))))
p2 = histogram!(c_lat,bins=0:0.05:0.7,normalize=:pdf,label="Hidden nodes",color=palette(:tab10)[4])
histogram(c_obs,bins=0:0.05:0.5,normalize=:pdf,xlabel=L"k'",ylabel=L"P(k')",label="Readout nodes",color=palette(:tab10)[1],yticks=([1:1:8;],string.(round.(1*0.05:1.0*0.05:8*0.05,digits=2))))
p2 = histogram!(c_lat,bins=0:0.05:0.5,normalize=:pdf,label="Hidden nodes",color=palette(:tab10)[4])


# Calculate average path legnth and clustering coefficient
function graph_propertys_eval(args, runs=1)
    L=[];C=[]

    for i in 1:runs
        if isfile(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "prune_model_"*string(iteration)*".bson"))
            model, O = load_model(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(i), "prune_model_"*string(iteration)*".bson"))
            res_1, res_2, _, _,_, _ = graph_structure(model.W_mask)
            push!(L,res_1);push!(C,res_2);
        end
    end
    return L, C
end

L, C = graph_propertys_eval(args)


# Calculate for other graph structures
L_BA = []; C_BA = []; L_WS = []; C_WS = []; L_ER = []; C_ER = [];
K_BA = 2; K_WS = 4; N = 50; n = 100

for i in 1:n
    a,b,_,_,_,_ = NetworkTopology.graph_structure(NetworkTopology.barabasi_albert_graph(N,K_BA,1,1))
    push!(L_BA,a); push!(C_BA,b)
    a,b,_,_,_,_ = NetworkTopology.graph_structure(NetworkTopology.erdos_renyi_graph(N,K_BA,1,1))
    push!(L_ER,a); push!(C_ER,b)
    a,b,_,_,_,_ = NetworkTopology.graph_structure(NetworkTopology.watts_strogatz_graph(N,K_WS,0.15,1))
    push!(L_WS,a); push!(C_WS,b)
end

#Plotting
p3 = boxplot([C,C_WS,C_BA,C_ER],outliers=false,
    lw=5,ylabel=L"C",
    palette=reverse(palette(:grayC10)[3:1:8]),
    xticks=([1,2,3,4],["Geomet.\npruned","WS","BA","ER"]),
    tickfontsize=18, labelfontsize=18,legendfontsize=18,size=(700,400),margin=5Plots.mm,legend=false
)
p4 = boxplot([L,L_WS,L_BA,L_ER],outliers=false,
    lw=5,ylabel=L"L",
    palette=reverse(palette(:grayC10)[3:1:8]),
    xticks=([1,2,3,4],["Geomet.\npruned","WS","BA","ER"]),
    tickfontsize=18, labelfontsize=18,legendfontsize=18,size=(700,400),margin=5Plots.mm,legend=false
)


#Plot all
plot(p1,p2,p4,p3,size=(1600,1000),tickfontsize=20, labelfontsize=34,legendfontsize=18,margins=8Plots.mm,layout = @layout[a{0.5h} b;c d])



