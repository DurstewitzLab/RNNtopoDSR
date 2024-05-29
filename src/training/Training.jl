module Training

using ..Utilities
export AbstractTFRecur, 
    TFRecur,
    WeakTFRecur,
    init_state!,
    force,
    train_!,
    sample_batch,
    sample_sequence,
    AbstractDataset,
    Dataset,
    Progress,
    update!,
    print_progress

include("dataset.jl")
include("forcing.jl")
include("tfrecur.jl")
include("progress.jl")
include("training_plrnn.jl")

end
