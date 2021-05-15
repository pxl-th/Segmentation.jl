# using CUDA
# CUDA.allowscalar(false)
using Zygote

function grad_acc()
    # device = gpu
    # conv = Conv((3, 3), 3=>4) |> device
    # x = randn(Float32, 10, 10, 3, 2) |> device
    # y = randn(Float32, 8, 8, 4, 2) |> device
    # θ = conv |> params

    # g1 = gradient(()->Flux.mse(conv(x), y), θ)
    # g2 = gradient(()->Flux.mse(conv(x), y), θ)

    w = randn(Float32, 2)
    x = randn(Float32, 2)
    p = Params([w])

    g1 = gradient(() -> sum(w .* x) / 2, p)
    g2 = gradient(() -> sum(w .* x) / 2, p)

    @info g1[w]
    @info g2[w]
    @info g1[w] + g2[w]
end
grad_acc()
