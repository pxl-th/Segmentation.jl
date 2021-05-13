module Segmentation

using Random
using Images
using DataLoaders
using Interpolations
using ProgressMeter

using CUDA
CUDA.allowscalar(false)
using Flux
using ResNet
using ParameterSchedulers: Scheduler, Cos

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

function get_pb(n, desc::String)
    Progress(
        n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white,
    )
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
    for f in readlines(path)[1:100]
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
    color_palette = [RGB{N0f8}((p ./ 255)...) for p in palette]
    bw_palette = color_palette .|> Gray{N0f8}
    color_palette, bw_palette
end

function mask_to_classes(mask, palette)
    classes = Array{Int64}(undef, size(mask))
    for (i, p) in palette |> enumerate
        classes[mask .== p] .= i
    end
    classes = Flux.onehotbatch(classes, 1:length(palette))
    permutedims(classes, (2, 3, 1))
end

"""
classes: (W, H, C)
"""
function probs_to_mask(classes, palette::Vector{T}) where T
    width, height = size(classes)[1:2]
    mask = Array{T}(undef, width, height)

    for i in 1:width, j in 1:height
        @inbounds mask[i, j] = palette[argmax(classes[i, j, :])]
    end
    mask
end

function main()
    # TODO image augmentation
    # TODO accumulate grads
    # TODO add ability to disable batchnorm

    device = gpu
    epochs = 100
    batch_size = 2
    color_palette, bw_palette = get_palette()
    classes = bw_palette |> length
    # resolution = (18 * 32, 14 * 32)
    resolution = (10 * 32, 7 * 32)

    base_dir = raw"C:\Users\tonys\projects\comma10k"
    train_files, valid_files = load_file_names(
        joinpath(base_dir, "files_trainable"),
    )

    valid_dataset = Dataset(base_dir, valid_files, resolution, bw_palette)
    valid_loader = DataLoader(valid_dataset, batch_size)

    λ1 = 1e-4
    λ0 = λ1 / 50
    period = epochs * length(train_files) ÷ batch_size
    optimizer = Scheduler(Cos(;λ0, λ1, period), ADAM())

    model = UNet(;
        classes, encoder=ResNetModel(;size=18, in_channels=3, classes=nothing),
    ) |> device
    θ = model |> params

    @info "Image resolution: $resolution [width, height]"
    @info "Train images: $(length(train_files))"
    @info "Validation images: $(length(valid_files))"
    @info "Total Parameters: $(length(θ))"
    @info "LR Scheduler: [λ0=$λ0, λ1=$λ1], period=$period"

    @info "--- Training ---"
    for epoch in 1:epochs
        train_loader = DataLoader(
            Dataset(base_dir, shuffle!(train_files), resolution, bw_palette),
            batch_size,
        )

        model |> trainmode!
        train_loss = 0.0
        bar = get_pb(length(train_loader), "[epoch $epoch | training]: ")
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            grads = gradient(θ) do
                loss = Flux.logitcrossentropy(x |> model, y; dims=3)
                train_loss += loss
                loss
            end

            Flux.Optimise.update!(optimizer, θ, grads)
            GC.gc()
            bar |> next!
        end
        train_loss /= length(train_loader)
        @info "epoch $epoch | train loss $train_loss | lr $(optimizer.optim.eta)"

        model |> testmode!
        bar = get_pb(length(valid_loader), "[epoch $epoch | testing]: ")
        validation_loss = 0.0
        for (i, (x, y)) in enumerate(valid_loader)
            x, y = x |> device, y |> device
            o = x |> model
            validation_loss += Flux.logitcrossentropy(o, y; dims=3)

            if i == 1
                o = softmax(o; dims=3) |> cpu
                pred = probs_to_mask(o[:, :, :, 1], color_palette)
                save("./pred-$epoch.png", permutedims(pred, (2, 1)))

                y = y |> cpu
                pred = probs_to_mask(y[:, :, :, 1], color_palette)
                save("./mask-$epoch.png", permutedims(pred, (2, 1)))
            end

            GC.gc()
            bar |> next!
        end
        validation_loss /= length(valid_loader)
        @info "epoch $epoch | validation loss $validation_loss"
    end
end
main()

end
