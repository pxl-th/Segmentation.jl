# Segmentation.jl

Segmentation models.

![](https://github.com/pxl-th/Segmentation.jl/blob/master/res/output.gif?raw=true)

## Install

```
]add https://github.com/pxl-th/Segmentation.jl.git
```

## Usage

```julia
classes = 5
encoder = EfficientNet.from_pretrained("efficientnet-b0"; include_head=false)
encoder_channels = EfficientNet.stages_channels(encoder)
model = UNet(;classes, encoder, encoder_channels)
mask = x |> model
```

## Supported architectures

- UNet

## Supported encoders

- [ResNet.jl](https://github.com/pxl-th/ResNet.jl)
- [EfficientNet.jl](https://github.com/pxl-th/EfficientNet.jl)

## Examples

See [example/](https://github.com/pxl-th/Segmentation.jl/tree/master/example) folder
on how to train on [comma10k](https://github.com/commaai/comma10k) dataset.
