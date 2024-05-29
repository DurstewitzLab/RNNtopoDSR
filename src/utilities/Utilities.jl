module Utilities

using LinearAlgebra
using Plots

const Maybe{T} = Union{T, Nothing}

export Maybe,
    uniform,
    randn_like,
    num_params,
    add_gaussian_noise,
    add_gaussian_noise!,
    create_folder_structure,
    store_hypers,
    load_defaults,
    load_json_f32,
    save_model,
    load_model,
    evaluate_Dstsp,
    evaluate_PE,
    evaluate_PSE,
    format_run_ID,
    check_for_NaNs,
    plot_reconstruction,
    watts_strogatz_graph,
    barabasi_albert_graph,
    erdos_renyi_graph,
    GeoHub_graph,
    graph_structure,
    SWI,
    lattice_reference,
    random_reference,
    clustering_coefficient,
    magnitude,
    geometry,
    random

include("helpers.jl")
include("utils.jl")
include("plotting.jl")
include("pruning.jl")

end