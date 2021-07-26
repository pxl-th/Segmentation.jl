using Flux

function encode(encoder, x)
    features = typeof(x)[]
    for block in encoder
        x = block(x)
        push!(features, x)
    end
    features
end

function main()
    device = gpu
    x = randn(Float32, 10, 10, 3, 1) |> device

    encoder = Chain(BatchNorm(3, identity), BatchNorm(3, identity)) |> device |> trainmode!
    # encoder = Chain(
    #     Conv((3, 3), 3=>3, identity; pad=SamePad()),
    #     Conv((3, 3), 3=>3, identity; pad=SamePad()),
    # ) |> device |> trainmode!
    θ = params(encoder)

    gradient(θ) do
        sum(reduce(+, encode(encoder, x)))
    end
end
main()
