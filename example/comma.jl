using Random
using LinearAlgebra

using Flux
using EfficientNet
# using ResNet
using Segmentation

using Augmentations
using DataLoaders
using Images
using Interpolations

using BSON: @save, @load, bson
using ProgressMeter
using VideoIO

Random.seed!(1)

include("data.jl")

function train_step(model, θ, x, y, optimizer, bar)
    loss_cpu = 0f0
    ∇ = gradient(θ) do
        l = Flux.logitcrossentropy(x |> model, y; dims=3)
        loss_cpu = l |> cpu
        l
    end

    Flux.Optimise.update!(optimizer, θ, ∇)
    bar |> next!
    loss_cpu
end

function valid_step(model, x, y, epoch, i, palette, bar)
    o = x |> model
    validation_loss = Flux.logitcrossentropy(o, y; dims=3) |> cpu

    if i ≤ 5
        o = softmax(o; dims=3) |> cpu
        pred = probs_to_mask(o[:, :, :, 1], palette)
        save("./pred-$epoch-$i.png", permutedims(pred, (2, 1)))

        y = y |> cpu
        pred = probs_to_mask(y[:, :, :, 1], palette)
        save("./mask-$epoch-$i.png", permutedims(pred, (2, 1)))
    end

    bar |> next!
    validation_loss
end

function main()
    device = gpu
    epochs = 1000
    batch_size = 2
    color_palette, bw_palette = get_palette()
    classes = bw_palette |> length
    resolution = (7 * 32, 10 * 32)

    flip_augmentation = FlipX(0.5)
    augmentations = Sequential([
        OneOf(0.5, [CLAHE(;p=1, rblocks=2, cblocks=2), Equalize(1)]),
        OneOf(0.25, [ToGray(1), Downscale(1, (0.25, 0.75))]),
        OneOf(0.5, [
            Blur(;p=1),
            GaussNoise(;p=1, σ_range=(0.03, 0.08)),
            RandomGamma(1, (0.5, 3)),
            RandomBrightness(1, 0.2),
        ]),
    ])

    base_dir = "/home/pxl-th/projects/comma10k"
    files = readlines(joinpath(base_dir, "files_trainable"))
    train_files, valid_files = load_file_names(files)

    train_dataset = Dataset(
        base_dir, train_files, resolution, bw_palette,
        flip_augmentation, augmentations,
    )
    valid_dataset = Dataset(base_dir, valid_files, resolution, bw_palette)
    valid_loader = DataLoader(valid_dataset, batch_size)

    optimizer = ADAM(3f-4)
    encoder = EfficientNet.from_pretrained("efficientnet-b0"; include_head=false)
    encoder_channels = EfficientNet.stages_channels(encoder)
    # encoder = ResNet.from_pretrained(18; classes=nothing)
    # encoder_channels = ResNet.stages_channels(encoder)
    model = UNet(;classes, encoder, encoder_channels) |> device
    θ = model |> params

    @info "Image resolution: $resolution [height, width]"
    @info "Train images: $(length(train_files))"
    @info "Validation images: $(length(valid_files))"
    @info "Total Parameters: $(length(θ))"
    @info "--- Training ---"

    for epoch in 1:epochs
        model |> trainmode!
        train_dataset.files |> shuffle!
        train_loader = DataLoader(train_dataset, batch_size)
        bar = get_pb(length(train_loader), "[epoch $epoch | training]: ")

        train_loss = 0f0
        for (x, y) in train_loader
            train_loss += train_step(
                model, θ, x |> device, y |> device, optimizer, bar,
            )
        end
        train_loss /= length(train_loader)
        @info "Epoch $epoch | Train Loss $train_loss"

        model |> testmode!
        bar = get_pb(length(valid_loader), "[epoch $epoch | testing]: ")
        validation_loss = 0f0
        for (i, (x, y)) in enumerate(valid_loader)
            validation_loss += valid_step(
                model, x |> device, y |> device,
                epoch, i, color_palette, bar,
            )
        end
        validation_loss /= length(valid_loader)
        @info "Epoch $epoch | Validation Loss $validation_loss"

        model_host = model |> cpu
        @save "./weights/v1/epoch-$epoch-valloss-$validation_loss.bson" model_host
    end
end

function eval()
    device = gpu
    iresolution = (7 * 32, 10 * 32)
    oresolution = (874, 1164)
    wresolution = (874 ÷ 3 + 1, 2 * oresolution[2] ÷ 3)
    @show wresolution

    μ = reshape([0.485f0, 0.456f0, 0.406f0], (3, 1, 1))
    σ = reshape([0.229f0, 0.224f0, 0.225f0], (3, 1, 1))
    color_palette, _ = get_palette()

    @load "./weights/v1/epoch-5-valloss-0.103507854.bson" model_host
    model = model_host |> device |> testmode!

    video_file = "/home/pxl-th/projects/SLAM.jl/data/5.hevc"
    reader = video_file |> openvideo

    i = 1
    open_video_out("5-seg.mp4", RGB{N0f8}, wresolution; framerate=25) do writer
    for frame in reader
        in_frame = imresize(frame, iresolution)
        in_frame = in_frame |> channelview .|> Float32
        in_frame .= (in_frame .- μ) ./ σ # ImageNet preprocessing.
        in_frame = permutedims(in_frame, (3, 2, 1))
        in_frame = Flux.unsqueeze(in_frame, 4) |> device

        probs = softmax(in_frame |> model; dims=3) |> cpu
        mask = probs_to_mask(probs[:, :, :, 1], color_palette)
        mask = permutedims(mask, (2, 1))

        wframe = hcat(imresize(frame, oresolution), imresize(mask, oresolution))
        wframe = imresize(wframe, wresolution)
        write(writer, wframe)

        @info i
        i += 1
    end
    end
    reader |> close
end

# main()
# eval()
