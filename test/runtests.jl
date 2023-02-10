module GridapBatteriesTests

using Test

@time @testset "PoissonAgFEM" begin include("PoissonAgFEMTests.jl") end

@time @testset "TransientTrimaterialPoissonAgFEM" begin include("TransientTrimaterialPoissonAgFEMTests.jl") end

end