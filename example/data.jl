get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white,
)

struct Dataset
    base::String
    files::Vector{String}
    resolution::Tuple{Int64, Int64} # (height, width)
    palette::Vector{Gray}

    flip_augmentation::Union{Augmentations.FlipX, Nothing}
    augmentations::Union{Augmentations.Sequential, Nothing}

    Dataset(
        base, files, resolution, palette,
        flip_augmentation = nothing, augmentations = nothing,
    ) = new(base, files, resolution, palette, flip_augmentation, augmentations)
end

DataLoaders.nobs(d::Dataset) = d.files |> length
function DataLoaders.getobs(d::Dataset, i::Int64)
    name = d.files[i]

    image = joinpath(d.base, "imgs", name) |> load .|> RGB
    mask = joinpath(d.base, "masks", name) |> load .|> RGB

    image = imresize(image, d.resolution)
    mask = imresize(mask, d.resolution; method=Constant())

    if d.flip_augmentation ≢ nothing
        image, mask = d.flip_augmentation([image, mask])
    end
    if d.augmentations ≢ nothing
        image = d.augmentations([image])[1]
    end

    image = image |> channelview .|> Float32
    # ImageNet preprocessing.
    μ = reshape([0.485f0, 0.456f0, 0.406f0], (3, 1, 1))
    σ = reshape([0.229f0, 0.224f0, 0.225f0], (3, 1, 1))
    image .= (image .- μ) ./ σ
    image = permutedims(image, (3, 2, 1))

    mask = mask .|> Gray
    mask = permutedims(mask, (2, 1))
    mask = mask_to_classes(mask, d.palette)

    image, mask
end

function load_file_names(files)
    train = String[]
    valid = String[]
    for f in files
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
        # [0, 0, 0]        # for padding
    ]
    color_palette = [RGB{N0f8}((p ./ 255f0)...) for p in palette]
    bw_palette = color_palette .|> Gray{N0f8}
    color_palette, bw_palette
end

"""
mask: Gray(W, H)
"""
function mask_to_classes(mask, palette)
    classes = Array{Int64}(undef, size(mask))
    for (i, p) in palette |> enumerate
        @inbounds classes[mask .== p] .= i
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
        @inbounds mask[i, j] = palette[argmax(@view(classes[i, j, :]))]
    end
    mask
end
