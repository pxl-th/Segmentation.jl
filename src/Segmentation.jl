module Segmentation
export UNet

using Flux
using ResNet

include("decoder.jl")
include("head.jl")

struct UNet
    encoder
    decoder
    segmentation
end
Flux.@functor UNet

function UNet(;
    classes, encoder, decoder_channels = (256, 128, 64, 32, 16),
)
    UNet(
        encoder,
        UNetDecoder(ResNet.stages_channels(encoder), decoder_channels),
        SegmentationHead((3, 3), decoder_channels[end]=>classes),
    )
end

(u::UNet)(x) = u.encoder(x, Val(:stages)) |> u.decoder |> u.segmentation

end
