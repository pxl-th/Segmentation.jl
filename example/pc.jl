using PackageCompiler

create_sysimage(
    [
        :Random,
        :LinearAlgebra,
        :CUDA,
        :Flux,
        :DataLoaders,
        :Images,
        :Interpolations,
        :BSON,
        :ProgressMeter,
    ]; sysimage_path="sys-comma.so", precompile_execution_file="example/precompile.jl",
)

