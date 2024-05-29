using Distributed
using Distributions
using Statistics
using Flux
using LinearAlgebra
using JLD
using ..Training
using ..Utilities

mutable struct Argument
    name::String
    value::Any
    id_str::String

    function Argument(name::String, value)
        if value isa Vector
            error("Please add an identifier name \
                  for arguments that are subject to grid search!")
        end
        return new(name, value, "")
    end
    Argument(name::String, value, id_str::String) = new(name, value, id_str)
end

# type aliases
ArgVec = Vector{Argument}
ArgDict = Dict{String, Any}
TaskVec = Vector{ArgDict}

add_to_name(name::String, arg::Argument) = name * "-" * arg.id_str * "_" * string(arg.value)

function prepare_base_name(default_name::String, args::ArgVec)::String
    name = default_name
    # filter name argument if specified by user
    arg = filter(arg -> arg.name == "name", args)
    if !isempty(arg)
        name = arg[1].value
    end
    return name
end

function check_arguments(defaults::ArgDict, args::ArgVec)
    for arg in args
        # check if arg exists
        @assert haskey(defaults, arg.name) "Argument/Setting <$(arg.name)> does not exist."

        # cast to correct type
        arg.value = arg.value .|> typeof(defaults[arg.name])
    end
end

function prepare_tasks(defaults::ArgDict, args::ArgVec, n_runs::Int)
    # check if arguments passed actually exist in default settings
    check_arguments(defaults, args)

    # extract multitasking name
    name = prepare_base_name(defaults["name"], args)

    # split arguments into the ones that are subject to 
    # undergo grid search and the ones constant
    const_args = filter(arg -> !(arg.value isa Vector), args)
    gs_args = filter(arg -> arg.value isa Vector, args)

    # overwrite default args with const args
    baseline_args = copy(defaults)
    for arg in const_args
        baseline_args[arg.name] = arg.value
        name = isempty(arg.id_str) ? name : add_to_name(name, arg)
    end
    baseline_args["name"] = name

    # done here, if no gs is performed
    tasks = [baseline_args]
    if !isempty(gs_args)
        tasks = generate_grid_search_tasks(baseline_args, gs_args)
    end

    # add multiple runs per task
    tasks = add_runs_to_tasks(tasks, n_runs)

    return tasks
end

function add_runs_to_tasks(tasks::TaskVec, n_runs::Int)
    tasks_w_runs = TaskVec()
    for task in tasks
        for r = 1:n_runs
            task_cp = copy(task)
            task_cp["run"] = r
            push!(tasks_w_runs, task_cp)
        end
    end
    @assert length(tasks_w_runs) == length(tasks) * n_runs
    return tasks_w_runs
end

function generate_grid_search_tasks(args::ArgDict, gs_args::ArgVec)
    # initialize with first gs variable
    tasks = TaskVec()
    init_arg = gs_args[1]
    add_values_to_task!(tasks, args, init_arg)

    # loop over other variables
    for arg in gs_args[2:end]
        new_tasks = copy(tasks)
        for task in tasks
            add_values_to_task!(new_tasks, task, arg)
        end
        # keep "mix terms"
        tasks = new_tasks[length(tasks)+1:end]
    end
    return tasks
end

function replace_arg(args::ArgDict, arg::Argument)
    args_cp = copy(args)
    args_cp[arg.name] = arg.value
    args_cp["name"] = add_to_name(args_cp["name"], arg)
    return args_cp
end

function add_values_to_task!(tasks::TaskVec, task::ArgDict, arg::Argument)
    for v in arg.value
        push!(tasks, replace_arg(task, Argument(arg.name, v, arg.id_str)))
    end
end



"""
    main_routine(args)

Function executed by every worker process.
"""
function main_routine(args::AbstractDict)
    # num threads
    n_threads = Threads.nthreads()
    BLAS.set_num_threads(n_threads)
    println("Running on $n_threads Thread(s)")

    # Initialize Dataset
    println("Initializing Dataset.")
    D = NetworkTopology.Training.Dataset(args["path_to_data"])

    # initialize plrnn model with wanted structure
    plrnn = initialize_model(args, D)  # model
    O = initialize_observation_model(args, D)  # Observation model

    #Introduce structure in weight matrix
    if args["small_world"] == "custom" && isfile(joinpath(args["initial_mask_path"],args["initial_mask_run"],"prune_model_"*string(args["initial_mask_ind"])*".bson"))
        m_temp, _ = load_model(joinpath(args["initial_mask_path"],args["initial_mask_run"],"prune_model_"*string(args["initial_mask_ind"])*".bson"))
        plrnn.W_mask = copy(m_temp.W_mask)
        if isfile(args["initial_mask_params"])
            m_temp, _ = load_model(args["initial_mask_params"])
            plrnn.W = copy(m_temp.W); plrnn.A = copy(m_temp.A); plrnn.h = copy(m_temp.h)
            println("Using custom mask strucutre with ",sum(plrnn.W_mask)," trainable parameters and initial parameters")
        else
            println("Using custom mask strucutre with ",sum(plrnn.W_mask)," trainable parameters")
        end
    elseif args["small_world"] != ""
        graph = getfield(Main, Symbol(args["small_world"]*"_graph"))
        plrnn.W_mask = copy(graph(args["latent_dim"], args["K"], args["p"], size(D.X)[2]))
        println("Using "*args["small_world"]*" graph structure with ",sum(plrnn.W_mask)," trainable parameters")
    end
    plrnn.W .= plrnn.W .* plrnn.W_mask

    # optimizer
    opt = initialize_optimizer(args)

    # create directories
    save_path = create_folder_structure(args["experiment"], args["name"], args["run"])

    # Save non trained model
    save_model([plrnn, O], joinpath(save_path, "checkpoints", "model_0.bson"),)

    #Calculate structre measures in weight matrix
    if hasproperty(plrnn, :W)
        L, C, Deg = graph_structure(plrnn.W)
        save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "graph_metrics.jld"), "graph_metrics", [L, C, Deg])
        println("Structre properties (L, C, Deg): ",round.([L, C, Deg],digits=3))    
    end

    # store hypers
    store_hypers(args, save_path)

    train_!(plrnn, O, D, opt, args, save_path)


    if args["prune_steps"] > 0 && args["model"] == "PLRNN"
        pruning_metrics = []

        for i in 1:args["prune_steps"]
            #Save previous results
            metrics = JLD.load(joinpath(pwd(), "Results", args["experiment"], args["name"], NetworkTopology.Utilities.format_run_ID(args["run"]), "metrics.jld"))["metrics"]  
            best_id = 1
            if length(metrics) > 1
                best_id = argmin(Matrix(mapreduce(permutedims,vcat, metrics)'),dims=2)[2][2] #Get best model 
                push!(pruning_metrics, metrics[best_id]) 
            else 
                push!(pruning_metrics, [NaN,NaN,NaN,NaN]) 
            end
            plrnn, O = load_model(joinpath("Results", args["experiment"], args["name"], NetworkTopology.Utilities.format_run_ID(args["run"]), "checkpoints", "model_"*string(best_id*args["scalar_saving_interval"])*".bson"))
            save_model([plrnn, O], joinpath(save_path, "prune_model_"*string(i)*".bson"),)
            save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "prune_metrics.jld"), "prune_metrics", pruning_metrics)

            #Pruning procedure
            pruning_measure = getfield(Main, Symbol(args["prune_measure"]))
            mask = pruning_measure(args, plrnn, O, D)
            println("Using "*args["prune_measure"]*" pruning")

            plrnn, O = load_model(joinpath("Results", args["experiment"], args["name"], NetworkTopology.Utilities.format_run_ID(args["run"]), "checkpoints", "model_0.bson"))
            plrnn.W_mask = mask
            plrnn.W .= plrnn.W .* mask

            #Graph metrics
            L, C, Deg = graph_structure(plrnn.W)
            old_graph_metrics = JLD.load(joinpath(pwd(), "Results", args["experiment"], args["name"], NetworkTopology.Utilities.format_run_ID(args["run"]), "graph_metrics.jld"))["graph_metrics"]
            JLD.save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "graph_metrics.jld"), "graph_metrics", push!(old_graph_metrics, L, C, Deg))
            println("Structre properties (L, C, Deg): ",round.([L, C, Deg],digits=3))

            #Training
            opt = initialize_optimizer(args)
            train_!(plrnn, O, D, opt, args, save_path)

        end

        save(joinpath("Results", args["experiment"], args["name"], Utilities.format_run_ID(args["run"]), "prune_metrics.jld"), "prune_metrics", pruning_metrics)        

    end

end
