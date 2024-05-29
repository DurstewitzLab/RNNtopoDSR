# Optimal Recurrent Network Topologies for Dynamical Systems Reconstruction [ICML 2024]

# 1. Getting started
To install the package, clone the repostiory and `cd` into the project folder:
Install the package in a new Julia environment:
```
julia> ]
(@v1.9) pkg> activate .
pkg> instantiate
```

# 2. Running the code
## 2.1 Single runs
To start a single training, execute the `main.jl` file, where arguments can be passed via command line. For example, to train a mcPLRNN with 50 latent dimensions for 1000 epochs using 4 threads, while keeping all other training parameters at their default setting, call
```
$ julia -t4 --project main.jl --model PLRNN --latent_dim 50 --epochs 1000
```
in your terminal of choice (bash/cmd). The [default settings](settings/defaults.json) can also be adjusted directly; one can then omit passing any arguments at the call site. The arguments are also listed in  in the [`argtable()`](src/parsing.jl) function.

## 2.2 Multiple runs + grid search
To run multiple trainings in parallel e.g. when grid searching hyperparameters, the `ubermain.jl` file is used. Currently, one has to adjust arguments which are supposed to differ from the [default settings](settings/defaults.json), and arguments that are supposed to be grid searched, in the `ubermain` function itself. This is as simple as adding an `Argument` to the `ArgVec` vector, which is passed the hyperparameter name (e.g. `latent_dim`), the desired value, and and identifier for discernibility and documentation purposes. If value is a vector of values, grid search for these hyperparameters is triggered. 
```Julia
function ubermain(n_runs::Int)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # list arguments here
    args = NetworkTopology.ArgVec([
        Argument("experiment", "Lorenz63-Pruning"),
        Argument("model", "PLRNN"),
        Argument("prune_measure", ["magnitude", "geometry", "random"], "prune"),
        Argument("prune_steps", 20),
        Argument("prune_value", 0.2)
    ])

    [...]
end
```
This will run a grid search over `prune_measure` using the `PLRNN`.

The identifier (e.g. `"prune"` in the snippet above) is only mandatory for arguments subject to grid search. Once Arguments are specified, call the ubermain file with the desired number of parallel worker proccesses (+ amount of threads per worker) and the number of runs per task/setting, e.g.
```{.sh}
$ julia -t2 --project ubermain.jl -p 10 -r 5
```
will queue 5 runs for each setting and use 10 parallel workers with each 2 threads.

## 2.3 Evaluating models
Evaluating trained model is done via `evaluate.jl`. Here, the path to the (test) data, the model experiment directory, and the settings to be passed to the various metrics employed, have to be provided.

# Specifics

## 1. Model architecture
For training the mean-centered PLRNN with identity mapping is used:
- mean-centred PLRNN &rarr; [`PLRNN`](src/models/plrnn.jl)
- identity mapping &rarr; [`Identity`](src/models/identity.jl)


## 2. Data Format
Data for the algorithm is expected to be a single trajectory in form of a $T \times N$ matrix (file format: `.npy`), where $T$ is the total number of time steps and $N$ is the data dimensionality. [Examples](example_data/) are provided.

## 3. Supported graph structures
These graph structres can be introduced in the parameter matrix of the PLRNN by setting `small_world` to:
- Watts-Strogatz model (`watts_strogatz`): Hyperparameters $K$, $p$
- Barabási-Albert model (`barabasi_albert`): Hyperparameters $K$
- Erdős–Rényi model (`erdos_renyi`): Hyperparameters $K$
- GeoHub model (`GeoHub`): Hyperparameters $K$
- Custom graph structure of a pruned PLRNN (`custom`): specify the PLRNN model containing the mask by setting `initial_mask_path` to the name path of the experiment, `initial_mask_run` to the run id and `initial_mask_ind` to the prune interation from which the mask is chosen. To use the models initial params set `initial_mask_params` to the respective model from which the parameters should be chosen.

## 4. Pruning
To apply iterative pruning to the PLRNN one needs to set the model to &rarr; [`PLRNN`](src/models/plrnn.jl) with &rarr; [`Identity`](src/models/identity.jl) mapping. The number of pruning steps is given by `prune_steps` and the amount of weights that get pruned every step by `prune_value`. The used pruning method applied can be specified by `prune_measure`, where one needs to choose between
- Geometry-based pruning (`geometry`)
- Magnitude-based pruning (`magnitude`)
- Random pruning (`random`)


# Versions
- >Julia 1.9
