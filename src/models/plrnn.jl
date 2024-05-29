using Flux: @functor
using Statistics: mean

using ..ObservationModels: ObservationModel, init_state

# abstract type
abstract type AbstractPLRNN end
(m::AbstractPLRNN)(z::AbstractVecOrMat) = step(m, z)
jacobian(m::AbstractPLRNN, z::AbstractVector) = Flux.jacobian(z -> m(z), z)[1]
jacobian(m::AbstractPLRNN, z::AbstractMatrix) = jacobian.([m], eachcol(z))


"""
mean-centered PLRNN
"""
mutable struct PLRNN{V <: AbstractVector, M <: AbstractMatrix} <: AbstractPLRNN
    A::V
    W::M
    W_mask::M
    h::V
end
@functor PLRNN (A, W, h)

# initialization/constructor
function PLRNN(M::Int)
    A, W, h = initialize_A_W_h(M)
    W_mask = Float32.(ones(M, M))
    return PLRNN(A, W, W_mask, h)
end

"""
    step(model, z)

Evolve `z` in time for one step according to the model `m` (equation).
`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.
"""
mean_center(z::AbstractVecOrMat) = z .- mean(z, dims = 1)
step(m::PLRNN, z::AbstractVecOrMat) = m.A .* z .+ (m.W .*  m.W_mask) * relu.(mean_center(z)) .+ m.h
function jacobian(m::PLRNN, z::AbstractVector)
    M, type = length(z), eltype(z)
    ℳ = type(1 / M) * (M * I - ones(type, M, M))
    return Diagonal(m.A) + (m.W .* m.W_mask) * Diagonal(ℳ * z .> 0) * ℳ
end




@inbounds """
    generate(model, z₁, T)

Generate a trajectory of length `T` using PLRNN model `m` given initial condition `z₁`.
"""
function generate(m::AbstractPLRNN, z₁::AbstractVector, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))
    # initial condition for model
    @views Z[1, :] .= z₁
    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= m(Z[t-1, :])
    end
    return Z
end

"""
    generate(model, observation_model, x₁, T)

Generate a trajectory of length `T` using PLRNN model `m` given initial condition `x₁`.
"""
function generate(m::AbstractPLRNN, obs::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(obs, x₁)
    ts = generate(m, z₁, T)
    return permutedims(obs(ts'), (2, 1))
end

