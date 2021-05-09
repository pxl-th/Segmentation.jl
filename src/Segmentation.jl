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

function UNet(;
    classes::Int64,
    encoder,
    decoder_channels::Vector{Int64} = [256, 128, 64, 64, 16],
)
    UNet(
        encoder,
        UNetDecoder(encoder.stages_channels, decoder_channels),
        SegmentationHead((3, 3), decoder_channels[end]=>classes),
    )
end

function (s::Seg)(x)
    # TODO classification
    s.encoder(x, Val(:stages)) |> s.decoder |> s.segmentation
end

function main()
    encoder = ResNetModel(;size=18, in_channels=3, classes=nothing)
    model = UNet(;classes=4, encoder)
    @info encoder.stages_channels

    x = randn(Float32, 224, 224, 3, 1)
    o = model(x)
    @info size(o)
end
main()

end
