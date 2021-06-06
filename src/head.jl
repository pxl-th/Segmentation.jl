function SegmentationHead(
    kernel::Tuple{Int64, Int64}, channels::Pair{Int64, Int64},
)
    Conv(kernel, channels; pad=kernel[1] ÷ 2)
    # TODO upsampling, activation
end
