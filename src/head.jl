function SegmentationHead(kernel, channels)
    Conv(kernel, channels; pad=kernel[1] ÷ 2)
    # TODO upsampling, activation
end
