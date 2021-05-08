module Segmentation

using Flux

include("decoder.jl")

"""
Segmentation inference:
    1. [features] = encoder(x)
    2. decoder_output = decoder([features]...)
    3. masks = segmentation_head(decoder_output)
    [4]. labels = classification_head([features][-1])
"""

"""
E - encoder
D - decoder
S - segmentation
C - classification
"""
# abstract type Seg{E, D, S, C} end
# struct UNet{E, D, S, C} <: Seg{E, D, S, C} end

d = DecoderBlock(4, 8, 12)

x = randn(Float32, 10, 10, 12, 1)
o = d(x)
@info size(o)

x = randn(Float32, 10, 10, 4, 1)
s = randn(Float32, 20, 20, 8, 1)
o = d(x, s)
@info size(o)


decoder = UNetDecoder(
    [3, 32, 24, 40, 112, 320],
    [256, 128, 64, 32, 16],
)

end
