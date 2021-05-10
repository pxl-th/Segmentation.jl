module Segmentation

using Flux
using ResNet

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
        UNetDecoder(ResNet.stages_channels(encoder), decoder_channels),
        SegmentationHead((3, 3), decoder_channels[end]=>classes),
    )
end

function (s::Seg)(x)
    # TODO classification
    s.encoder(x, Val(:stages)) |> s.decoder |> s.segmentation
end

function main()
    in_channels = 1
    encoder = ResNetModel(;size=18, in_channels, classes=nothing)
    @info stages_channels(encoder)
    model = UNet(;classes=4, encoder) |> gpu

    x = randn(Float32, 224, 224, in_channels, 1) |> gpu
    o = model(x)
    @info size(o), typeof(o)
end
main()

end
