using NetworkTopology
using LinearAlgebra


"""
    Magnitude-based pruning

Returns mask of a PLRNN model where smalles values are set to zero
Amount of pruned values specified in args["prune_value"]
"""

function magnitude(args, model, O, dataset)

    W = copy(model.W)
    n = Int(round(sum(abs.(W) .> 0) * args["prune_value"])) # Calculate absolute number of pruned values
    mask = model.W_mask
    W[W.==0] .= Inf # Already removed weights should not get pruned again

    for i in 1:n # Set mask entrys to zero with smallest weight
        ind = argmin(abs.(W))
        mask[ind] = 0
        W[ind] = Inf # prevent double pruning of weights
    end

    println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

    return mask
end


"""
    Geometry-based pruning

Returns mask of a PLRNN model where weights with lowest impact on geometrical reconstruction, measured by Dstsp, are set to zero
Amount of pruned values specified in args["prune_value"]
"""

# Calculate State space distance
function evaluate_Dstsp(args, model, O, dataset)
    
    X = dataset.X[1:30000,:]
    T = size(X, 1)
    z₁ = init_state(O, X[1, :])

    # generate trajectory and discard transients
    T̃ = floor(Int, 1.25 * T)
    Z = generate(model, z₁, T̃)[floor(Int, 0.25 * T)+1:end, :]
    X_gen = permutedims(O(Z'), (2, 1))

    # Dstsp
    if size(X[1, :])[1] > 11
        Dstsp = NetworkTopology.evaluate_Dstsp(X, X_gen, args["D_stsp_scaling"]) # For N>11 dims use GMM for calculating Dstsp
    else
        Dstsp = NetworkTopology.evaluate_Dstsp(X, X_gen, args["D_stsp_bins"])
    end

    return Dstsp
end


# Geometry-based pruning
function geometry(args, model, O, dataset)

    importance = zeros(size(model.W))
    reference = evaluate_Dstsp(args, model, O, dataset) # Calculate Dstsp reference value

    for (i,v) in enumerate(model.W) # For all weights calculate Dstsp difference
        if model.W[i] == 0 # Skip calculation for already pruned weights
            importance[i] = Inf 
            continue
        end
        model.W[i] = 0 # remove weight
        importance[i] = abs.(evaluate_Dstsp(args, model, O, dataset) - reference) # calculate Dstsp difference
        model.W[i] = v # reset weight
    end
    importance[isnan.(importance)] .= 20



    n = Int(round(sum(abs.(model.W) .> 0) * args["prune_value"])) # Calculate absolute number of pruned values
    mask = copy(model.W_mask)

    for i in 1:n # Set mask entrys to zero with smallest Dstsp influence
        ind = argmin(abs.(importance))
        mask[ind] = 0
        importance[ind] = Inf # prevent double pruning of weights
    end

    println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

    return mask

end


"""
    Random pruning

Returns mask of a PLRNN model where weights are set to zero by random
Amount of pruned values specified in args["prune_value"]
"""

function random(args, model, O, dataset)

    W = rand(size(model.W)[1],size(model.W)[2]) # Set up random matrix
    n = Int(round(sum(abs.(model.W) .> 0) * args["prune_value"])) # Calculate absolute number of pruned values
    mask = model.W_mask
    W[model.W .==0] .= Inf # Already removed weights should not get pruned again

    for i in 1:n # remove weights
        ind = argmin(abs.(W))
        mask[ind] = 0
        W[ind] = Inf # prevent double pruning of weights
    end

    println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

    return mask

end
