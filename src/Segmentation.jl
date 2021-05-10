module Segmentation

using Flux
using ResNet
using Images
using DataLoaders
using Interpolations

include("decoder.jl")
include("head.jl")

abstract type Seg{E, D, S} end
struct UNet{E, D, S} <: Seg{E, D, S}
    encoder::E
    decoder::D
    segmentation::S
end
Flux.@functor UNet

function UNet(;
    classes::Int64,
    encoder,
    decoder_channels::Vector{Int64} = [256, 128, 64, 32, 16],
)
    UNet(
        encoder,
        UNetDecoder(stages_channels(encoder), decoder_channels),
        SegmentationHead((3, 3), decoder_channels[end]=>classes),
    )
end

function (s::Seg)(x)
    # TODO classification
    s.encoder(x, Val(:stages)) |> s.decoder |> s.segmentation
end

struct Dataset
    base::String
    files::Vector{String}
    resolution::Tuple{Int64, Int64} # (width, height)
    palette::Vector{Gray}
end

DataLoaders.nobs(d::Dataset) = d.files |> length
function DataLoaders.getobs(d::Dataset, i::Int64)
    name = d.files[i]

    image = joinpath(d.base, "imgs", name) |> load |> channelview .|> Float32
    image = permutedims(image, (3, 2, 1))
    image = imresize(image, d.resolution)

    mask = joinpath(d.base, "masks", name) |> load .|> Gray
    mask = permutedims(mask, (2, 1))
    mask = imresize(mask, d.resolution; method=Constant())
    mask = mask_to_classes(mask, d.palette)

    image, mask
end

function load_file_names(path::String)
    train = String[]
    valid = String[]
    for f in path |> readlines
        push!(endswith(f, "9.png") ? valid : train, split(f, "masks/")[end])
    end
    train, valid
end

function get_palette()
    palette = [
        [128, 128, 96],  # undrivable
        [64, 32, 32],    # road
        [0, 255, 102],   # movable (vehicles, etc)
        [255, 0, 0],     # lane markings
        [204, 0, 255],   # my car
        [0, 0, 0]        # for padding
    ]
    [Gray{N0f8}(RGB{N0f8}((p ./ 255)...)) for p in palette]
end

function mask_to_classes(mask, palette)
    classes = Array{Int64}(undef, size(mask))
    for (i, p) in palette |> enumerate
        classes[mask .== p] .= i
    end
    classes = Flux.onehotbatch(classes, 1:length(palette))
    permutedims(classes, (2, 3, 1))
end

function main()
    epochs = 1
    batch_size = 1
    palette = get_palette()
    classes = palette |> length
    resolution = (18 * 32, 14 * 32)

    base_dir = raw"C:\Users\tonys\projects\comma10k"
    train_files, valid_files = load_file_names(
        joinpath(base_dir, "files_trainable"),
    )

    train_dataset = Dataset(base_dir, train_files, resolution, palette)
    train_loader = DataLoader(train_dataset, batch_size)
    valid_dataset = Dataset(base_dir, valid_files, resolution, palette)
    valid_loader = DataLoader(valid_dataset, batch_size)

    optimizer = ADAM()
    model = UNet(;
        classes, encoder=ResNetModel(;size=18, in_channels=3, classes=nothing),
    )
    trainables = model |> params

    for epoch in 1:epochs
        model |> trainmode!
        for (x, y) in train_loader
            grads = gradient(trainables) do
                Flux.logitcrossentropy(x |> model, y; dims=3)
            end
            Flux.Optimise.update!(optimizer, trainables, grads)
            break
        end

        model |> testmode!
        for (x, y) in valid_loader
            l = Flux.logitcrossentropy(x |> model, y; dims=3)
            println(l)
            break
        end
    end
end
main()

end
