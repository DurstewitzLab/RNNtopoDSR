using ArgParse
using Flux

function initialize_model(args::AbstractDict, D::AbstractDataset; mod = @__MODULE__)
    # gather args
    M = args["latent_dim"]

    model_name = args["model"]

    # model type in correct module scope
    model_t = @eval mod $(Symbol(model_name))

    # specify model args based on model type
    if model_t <: AbstractPLRNN
        model_args = (M,)
    end

    # optional arguments
    opt_args = args["optional_model_args"]

    # initialize model
    model = model_t(model_args..., opt_args...)

    println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end

function initialize_observation_model(args::AbstractDict, D::AbstractDataset)
    N = size(D.X, 2)
    M = args["latent_dim"]

    obs_model = Identity(N, M)

    println("Obs. Model / # Parameters: $(typeof(obs_model)) / $(num_params(obs_model))")
    return obs_model
end

function initialize_optimizer(args::Dict{String, Any})
    # optimizer chain
    opt_vec = []

    # vars
    κ = args["gradient_clipping_norm"]::Float32
    ηₛ = args["start_lr"]::Float32
    ηₑ = args["end_lr"]::Float32
    E = args["epochs"]::Int
    bpe = args["batches_per_epoch"]::Int

    # set gradient clipping
    if κ > zero(κ)
        push!(opt_vec, ClipNorm(κ))
    end

    # set SGD optimzier (ADAM, RADAM, etc)
    opt_sym = Symbol(args["optimizer"])
    opt = @eval $opt_sym($ηₛ)
    push!(opt_vec, opt)

    # set exponential decay learning rate scheduler
    γ = exp(log(ηₑ / ηₛ) / E)
    decay = ExpDecay(1, γ, bpe, ηₑ, 1)
    push!(opt_vec, decay)

    return Flux.Optimise.Optimiser(opt_vec...)
end



"""
    argtable()

Prepare the argument table holding the information of all possible arguments
and correct datatypes.
"""
function argtable()
    settings = ArgParseSettings()
    defaults = load_defaults()

    @add_arg_table settings begin
        # meta
        "--experiment"
        help = "The overall experiment name."
        arg_type = String
        default = defaults["experiment"] |> String

        "--name"
        help = "Name of a single experiment instance."
        arg_type = String
        default = defaults["name"] |> String

        "--run", "-r"
        help = "The run ID."
        arg_type = Int
        default = defaults["run"] |> Int

        "--scalar_saving_interval"
        help = "The interval at which scalar quantities are stored measured in epochs."
        arg_type = Int
        default = defaults["scalar_saving_interval"] |> Int

        "--image_saving_interval"
        help = "The interval at which images are stored measured in epochs."
        arg_type = Int
        default = defaults["image_saving_interval"] |> Int

        # data
        "--path_to_data", "-d"
        help = "Path to dataset used for training."
        arg_type = String
        default = defaults["path_to_data"] |> String

        # training
        "--teacher_forcing_interval"
        help = "The teacher forcing interval to use."
        arg_type = Int
        default = defaults["teacher_forcing_interval"] |> Int

        "--gaussian_noise_level"
        help = "Noise level of gaussian noise added to teacher signals."
        arg_type = Float32
        default = defaults["gaussian_noise_level"] |> Float32

        "--sequence_length", "-T"
        help = "Length of sequences sampled from the dataset during training."
        arg_type = Int
        default = defaults["sequence_length"] |> Int

        "--batch_size", "-S"
        help = "The number of sequences to pack into one batch."
        arg_type = Int
        default = defaults["batch_size"] |> Int

        "--epochs", "-e"
        help = "The number of epochs to train for."
        arg_type = Int
        default = defaults["epochs"] |> Int

        "--batches_per_epoch"
        help = "The number of batches processed in each epoch."
        arg_type = Int
        default = defaults["batches_per_epoch"] |> Int

        "--gradient_clipping_norm"
        help = "The norm at which to clip gradients during training."
        arg_type = Float32
        default = defaults["gradient_clipping_norm"] |> Float32

        "--optimizer"
        help = "The optimizer to use for SGD optimization. Must be one provided by Flux.jl."
        arg_type = String
        default = defaults["optimizer"] |> String

        "--start_lr"
        help = "Learning rate passed to the optimizer at the beginning of training."
        arg_type = Float32
        default = defaults["start_lr"] |> Float32

        "--end_lr"
        help = "Target learning rate at the end of training due to exponential decay."
        arg_type = Float32
        default = defaults["end_lr"] |> Float32

        # model
        "--model", "-m"
        help = "RNN to use."
        arg_type = String
        default = defaults["model"] |> String

        "--latent_dim", "-M"
        help = "RNN latent dimension."
        arg_type = Int
        default = defaults["latent_dim"] |> Int

        "--observation_model", "-o"
        help = "Observation model to use."
        arg_type = String
        default = defaults["observation_model"] |> String

        # Metrics
        "--D_stsp_scaling"
        help = "GMM scaling parameter."
        arg_type = Float32
        default = defaults["D_stsp_scaling"] |> Float32

        "--D_stsp_bins"
        help = "Number of bins for D_stsp binning method."
        arg_type = Int
        default = defaults["D_stsp_bins"] |> Int

        "--PSE_smoothing"
        help = "Gaussian kernel smoothing σ for power spectrum smoothing."
        arg_type = Float32
        default = defaults["PSE_smoothing"] |> Float32

        "--PE_n"
        help = "n-step ahead prediction error."
        arg_type = Int
        default = defaults["PE_n"] |> Int

        "--optional_model_args"
        help = "Optional model arguments."
        arg_type = Vector{String}
        default = defaults["optional_model_args"] |> Vector{String}

        #Pruning/Graph structure of weights

        "--small_world"
        help = "Graph structure used in PLRNN"
        arg_type = String
        default = defaults["small_world"] |> String

        "--K"
        help = "Degree Hyperparameter of graph"
        arg_type = Int
        default = defaults["K"] |> Int

        "--p"
        help = "Rewiring probability for Watts-Strogatz graph structure"
        arg_type = Float32
        default = defaults["p"] |> Float32

        "--prune_steps"
        help = "Number of pruning steps to be done"
        arg_type = Int
        default = defaults["prune_steps"] |> Int

        "--prune_value"
        help = "How many weights to prune"
        arg_type = Float32
        default = defaults["prune_value"] |> Float32

        "--prune_measure"
        help = "Which pruning measure to choose"
        arg_type = String
        default = defaults["prune_measure"] |> String

        "--initial_mask_path"
        help = "Initial mask path used for a model"
        arg_type = String
        default = defaults["initial_mask_path"] |> String

        "--initial_mask_ind"
        help = "Initial mask ind used for a model"
        arg_type = Int
        default = defaults["initial_mask_ind"] |> Int

        "--initial_mask_run"
        help = "Initial mask run number used for a model"
        arg_type = String
        default = defaults["initial_mask_run"] |> String

        "--initial_mask_params"
        help = "Initial mask params used for a model"
        arg_type = String
        default = defaults["initial_mask_params"] |> String


    end
    return settings
end

"""
    parse_commandline()

Parses all commandline arguments for execution of `main.jl`.
"""
parse_commandline() = parse_args(argtable())
