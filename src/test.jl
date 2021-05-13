using CUDA
CUDA.allowscalar(false)
using Zygote

function grad_acc()
    # device = gpu
    # conv = Conv((3, 3), 3=>4) |> device
    # x = randn(Float32, 10, 10, 3, 2) |> device
    # y = randn(Float32, 8, 8, 4, 2) |> device
    # θ = conv |> params

    # g1 = gradient(()->Flux.mse(conv(x), y), θ)
    # g2 = gradient(()->Flux.mse(conv(x), y), θ)

    w = randn(Float32, 2) |> cu
    x1 = randn(Float32, 2) |> cu
    x2 = randn(Float32, 2) |> cu
    p = Params([w])

    g1 = gradient(() -> sum(w .* x1), p)
    g2 = gradient(() -> sum(w .* x2), p)

    g1 .+= g2
end
grad_acc()
