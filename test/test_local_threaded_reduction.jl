using Test
using HaloArrays

@testset "Local and threaded reductions" begin
    local_u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
    local_v = similar(local_u)
    interior_view(local_u) .= [1, 2, 3, 4]
    interior_view(local_v) .= [10, 20, 30, 40]

    @test mapreduce(identity, +, local_u) == 10
    @test reduce(+, local_u) == 10
    @test mapfoldl(abs2, +, local_u) == 30
    @test mapfoldr(abs2, +, local_u) == 30
    @test mapreduce((x, y) -> x * y, +, local_u, local_v) == 300
    @test mapreduce(x -> x[1] * x[2], +, zip(local_u, local_v)) == 300
    @test all(x -> x > 0, local_u)
    @test any(x -> x == 3, local_u)

    local_fields = MultiHaloArray((; u=local_u, v=local_v))
    @test mapreduce(identity, +, local_fields) == 110
    @test mapfoldl(identity, +, local_fields) == 110
    @test mapfoldr(identity, +, local_fields) == 110
    @test all(x -> x > 0, local_fields)
    @test any(x -> x == 40, local_fields)

    threaded_u = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
    threaded_v = similar(threaded_u)
    interior_view(threaded_u, 1) .= [1, 2, 3]
    interior_view(threaded_u, 2) .= [4, 5, 6]
    interior_view(threaded_v, 1) .= [10, 20, 30]
    interior_view(threaded_v, 2) .= [40, 50, 60]

    @test mapreduce(identity, +, threaded_u) == 21
    @test reduce(+, threaded_u) == 21
    @test mapfoldl(abs2, +, threaded_u) == 91
    @test mapfoldr(abs2, +, threaded_u) == 91
    @test mapreduce((x, y) -> x * y, +, threaded_u, threaded_v) == 910
    @test mapreduce(x -> x[1] * x[2], +, zip(threaded_u, threaded_v)) == 910
    @test all(x -> x > 0, threaded_u)
    @test any(x -> x == 6, threaded_u)

    threaded_fields = MultiHaloArray((; u=threaded_u, v=threaded_v))
    @test mapreduce(identity, +, threaded_fields) == 231
    @test mapfoldl(identity, +, threaded_fields) == 231
    @test mapfoldr(identity, +, threaded_fields) == 231
    @test all(x -> x > 0, threaded_fields)
    @test any(x -> x == 60, threaded_fields)

    @test mapreduce(identity, +, threaded_u; dims=1)[] == 21
end
