module Segmentation
export UNet

using Flux

include("decoder.jl")
include("head.jl")

struct UNet
    encoder
    decoder
    segmentation
end
Flux.@functor UNet

function UNet(;
    classes, encoder, encoder_channels, decoder_channels = (256, 128, 64, 32, 16),
)
    UNet(
        encoder,
        UNetDecoder(encoder_channels, decoder_channels),
        SegmentationHead((3, 3), decoder_channels[end]=>classes),
    )
end

function (u::UNet)(x)
    features = u.encoder(x, Val(:stages))
    decoded = u.decoder(features)
    u.segmentation(decoded)
end

end
