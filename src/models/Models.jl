module Models

using Flux, LinearAlgebra

using ..Utilities

export AbstractPLRNN,
    PLRNN,
    generate,
    jacobian,
    uniform_init,
    gaussian_init

include("initialization.jl")
include("plrnn.jl")

end