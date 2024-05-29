using Flux
using Graphs
using SimpleWeightedGraphs
using LinearAlgebra
using Statistics
using StatsBase
using NaNStatistics
using BSON: load

num_params(m) = sum(length, Flux.params(m))

load_model(path::String;mod=@__MODULE__) = load(path, mod)[:model]

function evaluate_Dstsp(X, X_gen, bins_or_scaling)
    compute = bins_or_scaling > zero(bins_or_scaling)
    return compute ? state_space_distance(X, X_gen, bins_or_scaling) : missing
end

function evaluate_PSE(X, X_gen, smoothing)
    compute = smoothing > zero(smoothing)
    return compute ? power_spectrum_error(X, X_gen, smoothing) : missing
end

function evaluate_PE(m, O, X, n)
    compute = n > zero(n)
    return compute ? prediction_error(m, O, X, n) : missing
end

uniform(a, b) = rand(eltype(a)) * (b - a) + a
uniform(size, a, b) = rand(eltype(a), size) .* (b - a) .+ a

randn_like(X::AbstractArray{T, N}) where {T, N} = randn(T, size(X)...)

add_gaussian_noise!(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+= noise_level .* randn_like(X)
add_gaussian_noise(X::AbstractArray{T, N}, noise_level::T) where {T, N} =
    X .+ noise_level .* randn_like(X)



"""
Watts-Strogatz graph structure
"""

function watts_strogatz_graph(N, K, p, dims)

    G = Graphs.watts_strogatz(N, K, p)
    G_M = Matrix(Graphs.LinAlg.adjacency_matrix(G))

    return G_M
end

"""
Barabasi-albert graph structure
"""

function barabasi_albert_graph(N, K, p, dims)

    G_ini = complete_graph(K)
    G = Graphs.barabasi_albert!(G_ini, N, K)

    G_M = Matrix(Graphs.LinAlg.adjacency_matrix(G))

    return G_M
end

"""
Erdos-Renyi graph structure
"""

function erdos_renyi_graph(N, K, p, dims)

    G = Graphs.erdos_renyi(N, K*N)
    G_M = Matrix(Graphs.LinAlg.adjacency_matrix(G))

    return G_M
end

"""
GeoHub graph structure
"""

# Calculate connection probabilities
function graph_probs(G, K, dims, out = false)
    probs = ones(nv(G))
    for i in 1:nv(G)
        probs[i] = length(inneighbors(G,i)) + 0.05
    end
    if !out
        for i in 1:dims 
            probs[i]+=K
        end
    end
    if out 
        for i in 1:nv(G)
            probs[i] += length(outneighbors(G,i)) + 0.05
        end
        probs .+= K/2
    end
    return probs./sum(probs)
end


function GeoHub_graph(N, K, p, dims)
    K = Int(round(K/2))
    
    G = complete_graph(dims)
    G = SimpleDiGraph(G)
    add_vertices!(G,N-dims)

    #Generate ingoing connections
    for i in 1:N
        probs = graph_probs(G,K,dims)
        dsts = sample(1:N,Weights(probs),K,replace=false)
        for dst in dsts
            test_it = 1
            while (has_edge(G,i,dst) || dst==i) && test_it<(N*10) dst = sample(1:N,Weights(probs)); test_it+=1 end
            add_edge!(G,i,dst)
        end
    end

    #Generating outgoing connections
    for i in 1:N
        probs = graph_probs(G,K,dims,true)
        srcs = sample(1:N,Weights(probs),K,replace=false)
        for src in srcs
            test_it = 1
            while (has_edge(G,src,i) || src==i) && test_it<(N*10) src = sample(1:N,Weights(probs)); test_it+=1 end
            add_edge!(G,src,i)
        end
    end

    G_M = Matrix(Graphs.LinAlg.adjacency_matrix(G))
    G_M[diagind(G_M)[1:dims]] .= 1 #Allow non-linear self connection in readout nodes

    return G_M

end


"""
Weight structure calculation (measures L,C,Deg,Centralitys)
"""

function clustering_coefficient(Adj::AbstractMatrix)
    cs = []
    ks = []

    A = Adj .- Diagonal(Adj)
    for i in 1:size(A)[1]
        k_tot=sum(A[i,:])+sum(A[:,i])
        k_a=(A*A)[i,i]
        push!(ks,k_tot*(k_tot-1)-2*k_a)
    end

    for i in 1:size(A)[1]
        temp = 0
        for j in 1:size(A)[1]
            for k in 1:size(A)[1]
                if i==j || i==k || j==k continue end
                temp += (A[i,j]+A[j,i])*(A[j,k]+A[k,j])*(A[i,k]+A[k,i])
            end
        end
        push!(cs, 1/2 *temp / ks[i])
    end 

    cs[cs .== Inf] .= NaN
    return nanmean(Float64.(cs))
end

function graph_structure(adj_matrix)
    n = size(adj_matrix, 1)  # Get the number of nodes
    G = DiGraph(Int.(abs.(adj_matrix) .> 0))

    # Calculate metrics
    paths = Graphs.floyd_warshall_shortest_paths(G)
    n_access = length(paths.dists[(.!isinf.(paths.dists)) .* (paths.dists .< 10000)]) #number of accessible connections
    L = sum(paths.dists[(.!isinf.(paths.dists)) .* (paths.dists .< 10000)]) / (n_access-n) # subtract n for disregarding loops
    C = clustering_coefficient(abs.(adj_matrix) .> 0 )
    D = mean(Graphs.degree(G))

    return L,C,D
end

"""
Small world index
"""

function random_reference(G)
    
    n_edges = Int(round(length(edges(G))/2))
    n_nodes = length(vertices(G))

    G_ref = Graphs.erdos_renyi(n_nodes,n_edges)

    return Matrix(Graphs.adjacency_matrix(G_ref))
end

function lattice_reference(G)
    graph = Graphs.SimpleGraph(nv(G))

    v = 1
    run = 1
    for i in 1:Int(round((ne(G))/2))
        if v == nv(G)
            add_edge!(graph,v,run)
            run += 1
            v = 1
        elseif v >= nv(G) - run + 1
            add_edge!(graph,v,v+run-nv(G))
            v += 1
        else
            add_edge!(graph,v,v+run)
            v += 1
        end
    end

    return Int.(Matrix(Graphs.LinAlg.adjacency_matrix(graph)) .> 0)
end

function SWI(G_M)
    L,C,_,_,_,_=graph_structure(G_M)

    G = DiGraph(Int.(abs.(G_M) .> 0))

    L_R = []; L_L=[]; C_L=[]; C_R=[]
    for i in 1:10
        G_L = lattice_reference(G)
        G_R = random_reference(G)
        L_L_s,C_L_s,_,_,_,_=graph_structure(G_L)
        L_R_s,C_R_s,_,_,_,_=graph_structure(G_R)
        push!(L_R,L_R_s); push!(C_L,C_L_s); push!(L_L,L_L_s); push!(C_R,C_R_s)
    end
    L_R = mean(L_R); C_L = mean(C_L); L_L = mean(L_L); C_R = mean(C_R)

    return max(min((L-L_L)/(L_R-L_L) * ((C-C_R)/(C_L-C_R)),1),0)
end
