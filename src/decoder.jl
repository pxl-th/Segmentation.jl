conv2drelu(
    kernel::Tuple{Int64, Int64}, channels::Pair{Int64, Int64};
    pad::Int64 = 0, stride::Int64 = 1,
) = (
    Conv(kernel, channels; stride, pad, bias=false),
    BatchNorm(channels[2], relu),
)

struct DecoderBlock{C}
    conv::C
end
Flux.@functor DecoderBlock

function DecoderBlock(in_channels, skip_channels, out_channels)
    conv1 = conv2drelu(
        (3, 3), (in_channels + skip_channels)=>out_channels; pad=1,
    )
    conv2 = conv2drelu((3, 3), out_channels=>out_channels; pad=1)
    DecoderBlock(Chain(conv1..., conv2...))
end

function (d::DecoderBlock)(
    x::AbstractArray{T}, skip::Union{AbstractArray{T}, Nothing},
) where T
    o = upsample_nearest(x, (2, 2))
    if skip ≢ nothing
        o = cat(o, skip; dims=3)
    end
    o |> d.conv
end

struct UNetDecoder{B}
    blocks::B
end
Flux.@functor UNetDecoder

function UNetDecoder(encoder_channels, decoder_channels)
    encoder_channels = encoder_channels[end:-1:2]
    head_channels = encoder_channels[1]
    in_channels = [head_channels, decoder_channels[1:end - 1]...]
    skip_channels = [encoder_channels[2:end]..., 0]

    UNetDecoder([
        DecoderBlock(inc, sc, oc)
        for (inc, sc, oc) in zip(in_channels, skip_channels, decoder_channels)
    ])
end

function (d::UNetDecoder)(features)
    features = features[end:-1:2]
    head, skips = features[1], features[2:end]

    x = head
    for (i, block) in enumerate(d.blocks)
        skip = nothing
        if i ≤ length(skips)
            skip = skips[i]
        end
        x = block(x, skip)
    end
    x
end
