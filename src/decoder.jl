struct DecoderBlock{C, D}
    # TODO attention1
    # TODO attention2
    conv1::C
    conv2::D
end
Flux.@functor DecoderBlock

function DecoderBlock(in_channels, skip_channels, out_channels)
    conv1 = conv2drelu(
        (3, 3), (in_channels + skip_channels)=>out_channels; pad=1,
    )
    conv2 = conv2drelu((3, 3), out_channels=>out_channels; pad=1)
    DecoderBlock(conv1, conv2)
end

function (d::DecoderBlock)(
    x::AbstractArray{T}, skip::Union{AbstractArray{T}, Nothing} = nothing
) where T
    x = upsample_nearest(x, (2, 2))
    if skip ≢ nothing
        x = cat(x, skip; dims=3) # TODO |> attention1
    end
    x |> d.conv1 |> d.conv2 # TODO |> attention2
end

function conv2drelu(
    kernel::Tuple{Int64, Int64}, channels::Pair{Int64, Int64};
    pad::Int64 = 0, stride::Int64 = 1,
)
    Chain(
        Conv(kernel, channels; stride, pad, bias=false),
        BatchNorm(channels[2]), # bias=false because of batch norm.
        x -> x .|> relu,
    )
end

struct UNetDecoder{B}
    blocks::B
end
Flux.@functor UNetDecoder

function UNetDecoder(
    encoder_channels::Vector{Int64}, decoder_channels::Vector{Int64},
)
    encoder_channels = encoder_channels[end:-1:2]
    head_channels = encoder_channels[1]
    in_channels = [head_channels; decoder_channels[1:end - 1]]
    skip_channels = [encoder_channels[2:end]; 0]

    UNetDecoder([
        DecoderBlock(inc, sc, oc)
        for (inc, sc, oc)
        in zip(encoder_channels, skip_channels, decoder_channels)
    ])
end

function (d::UNetDecoder)(features)
    features = features[end:-1:2]
    head = features[1]
    skips = features[2:end]

    x = head
    for (i, block) in enumerate(d.blocks)
        x = block(x, i ≤ length(skips) ? skips[i] : nothing)
    end
    x
end
