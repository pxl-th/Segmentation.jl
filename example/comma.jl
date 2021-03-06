using Random
using LinearAlgebra

using Flux
using EfficientNet
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
        l = Flux.logitcrossentropy(model(x), y; dims=3)
        loss_cpu = cpu(l)
        l
    end
    Flux.Optimise.update!(optimizer, θ, ∇)

    next!(bar)
    loss_cpu
end

function valid_step(model, x, y, epoch, i, palette, bar)
    o = model(x)
    validation_loss = cpu(Flux.logitcrossentropy(o, y; dims=3))

    if i ≤ 5
        o = cpu(softmax(o; dims=3))
        pred = probs_to_mask(o[:, :, :, 1], palette)
        save("./pred-$epoch-$i.png", permutedims(pred, (2, 1)))

        y = cpu(y)
        pred = probs_to_mask(y[:, :, :, 1], palette)
        save("./mask-$epoch-$i.png", permutedims(pred, (2, 1)))
    end

    next!(bar)
    validation_loss
end

function main()
    transfer = gpu ∘ f32
    epochs = 1000
    batch_size = 2
    color_palette, bw_palette = get_palette()
    classes = length(bw_palette)
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

    base_dir = "/home/pxl-th/projects/datasets/comma10k"
    files = readlines(joinpath(base_dir, "files_trainable"))
    train_files, valid_files = load_file_names(files)

    train_dataset = Dataset(
        base_dir, train_files, resolution, bw_palette,
        flip_augmentation, augmentations)
    valid_dataset = Dataset(base_dir, valid_files, resolution, bw_palette)
    valid_loader = DataLoader(valid_dataset, batch_size)

    optimizer = ADAM(3f-4)
    encoder = EfficientNet.from_pretrained("efficientnet-b0"; include_head=false)
    encoder_channels = collect(encoder.stages_channels)
    model = transfer(UNet(;classes, encoder, encoder_channels))
    θ = params(model)

    @info "Image resolution: $resolution [height, width]"
    @info "Train images: $(length(train_files))"
    @info "Validation images: $(length(valid_files))"
    @info "Total Parameters: $(length(θ))"
    @info "--- Training ---"

    for epoch in 1:epochs
        trainmode!(model)
        shuffle!(train_dataset.files)
        train_loader = DataLoader(train_dataset, batch_size)

        bar = get_pb(length(train_loader), "[epoch $epoch | training]: ")
        train_loss = 0f0
        for (x, y) in train_loader
            train_loss += train_step(
                model, θ, transfer(x), transfer(y), optimizer, bar)
        end
        train_loss /= length(train_loader)
        @info "Epoch $epoch | Train Loss $train_loss"

        testmode!(model)
        bar = get_pb(length(valid_loader), "[epoch $epoch | testing]: ")
        validation_loss = 0f0
        for (i, (x, y)) in enumerate(valid_loader)
            validation_loss += valid_step(
                model, transfer(x), transfer(y),
                epoch, i, color_palette, bar)
        end
        validation_loss /= length(valid_loader)
        @info "Epoch $epoch | Validation Loss $validation_loss"

        model_host = cpu(model)
        @save "./weights/v1/epoch-$epoch-valloss-$validation_loss.bson" model_host
    end
end

function eval()
    transfer = gpu ∘ f32
    model_res = (7 * 32, 10 * 32)
    original_res = (874, 1164)
    write_res = (874 ÷ 3 + 1, 2 * original_res[2] ÷ 3)
    @show write_res

    μ = reshape([0.485f0, 0.456f0, 0.406f0], (3, 1, 1))
    σ = reshape([0.229f0, 0.224f0, 0.225f0], (3, 1, 1))
    color_palette, _ = get_palette()

    @load "./weights/v1/epoch-5-valloss-0.103507854.bson" model_host
    model = testmode!(transfer(model_host))

    output_video = "5-seg.mp4"
    video_file = "/home/pxl-th/projects/SLAM.jl/data/5.hevc"
    reader = openvideo(video_file)

    i = 1
    open_video_out(output_video, RGB{N0f8}, write_res; framerate=25) do writer
    for frame in reader
        in_frame = imresize(frame, model_res)
        in_frame = Float32.(channelview(in_frame))
        in_frame .= (in_frame .- μ) ./ σ # ImageNet preprocessing.
        in_frame = permutedims(in_frame, (3, 2, 1))
        in_frame = transfer(Flux.unsqueeze(in_frame, 4))

        probs = cpu(softmax(model(in_frame); dims=3))
        mask = probs_to_mask(probs[:, :, :, 1], color_palette)
        mask = permutedims(mask, (2, 1))

        wframe = hcat(imresize(frame, original_res), imresize(mask, original_res))
        wframe = imresize(wframe, write_res)
        write(writer, wframe)

        @info i
        i += 1
    end
    end
    close(reader)
end

main()
# eval()
