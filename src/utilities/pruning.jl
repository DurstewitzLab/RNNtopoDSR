using NetworkTopology
using LinearAlgebra


"""
Magnitude pruning
"""

function magnitude(args, model, O, dataset)

    if hasproperty(model, :W)
        W = copy(model.W)
        n = Int(round(sum(abs.(W) .> 0) * args["prune_value"]))
        mask = model.W_mask
        W[W.==0] .= Inf

        for i in 1:n
            ind = argmin(abs.(W))
            mask[ind] = 0
            W[ind] = Inf
        end

        println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

        return mask
    end
end


"""
Geometry-based pruning
"""

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
        Dstsp = NetworkTopology.evaluate_Dstsp(X, X_gen, args["D_stsp_scaling"])
    else
        Dstsp = NetworkTopology.evaluate_Dstsp(X, X_gen, args["D_stsp_bins"])
    end

    return Dstsp
end



function geometry(args, model, O, dataset)

    if hasproperty(model, :W)

        importance = zeros(size(model.W))
        reference = evaluate_Dstsp(args, model, O, dataset)

        for (i,v) in enumerate(model.W)
            if model.W[i] == 0
                importance[i] = Inf
                continue
            end
            model.W[i] = 0
            importance[i] = abs.(evaluate_Dstsp(args, model, O, dataset) - reference)
            model.W[i] = v
        end
        importance[isnan.(importance)] .= 20



        n = Int(round(sum(abs.(model.W) .> 0) * args["prune_value"]))
        mask = copy(model.W_mask)

        for i in 1:n
            ind = argmin(abs.(importance))
            mask[ind] = 0
            importance[ind] = Inf
        end

        println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

        return mask

    end

end


"""
Random pruning
"""

function random(args, model, O, dataset)

    if hasproperty(model, :W) 
        W = rand(size(model.W)[1],size(model.W)[2])
        n = Int(round(sum(abs.(model.W) .> 0) * args["prune_value"]))
        mask = model.W_mask
        W[model.W .==0] .= Inf

        for i in 1:n
            ind = argmin(abs.(W))
            mask[ind] = 0
            W[ind] = Inf
        end

        println("Pruning achived sparsity: ", 1 - sum(mask)/length(mask))

        return mask

    end
end
