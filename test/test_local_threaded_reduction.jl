using Test
using HaloArrays

@testset "Local and threaded reductions" begin
    local_u = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
    local_v = similar(local_u)
    interior_view(local_u) .= [1, 2, 3, 4]
    interior_view(local_v) .= [10, 20, 30, 40]

    @test mapreduce(identity, +, local_u) == 10
    @test reduce(+, local_u) == 10
    @test sum(local_u) == 10
    @test sum(abs2, local_u) == 30
    @test maximum(local_u) == 4
    @test minimum(local_u) == 1
    @test mapfoldl(abs2, +, local_u) == 30
    @test mapfoldr(abs2, +, local_u) == 30
    @test mapreduce((x, y) -> x * y, +, local_u, local_v) == 300
    @test mapreduce(x -> x[1] * x[2], +, zip(local_u, local_v)) == 300
    @test all(x -> x > 0, local_u)
    @test any(x -> x == 3, local_u)

    maybe_local = MaybeHaloArray(local_u)
    @test sum(maybe_local) == 10
    @test maximum(maybe_local) == 4
    @test minimum(maybe_local) == 1

    local_fields = MultiHaloArray((; u=local_u, v=local_v))
    @test mapreduce(identity, +, local_fields) == 110
    @test sum(local_fields) == 110
    @test maximum(local_fields) == 40
    @test minimum(local_fields) == 1
    @test mapfoldl(identity, +, local_fields) == 110
    @test mapfoldr(identity, +, local_fields) == 110
    @test all(x -> x > 0, local_fields)
    @test any(x -> x == 40, local_fields)

    local_array_fields = ArrayOfHaloArray([local_u, local_v])
    @test mapreduce(identity, +, local_array_fields) == 110
    @test sum(local_array_fields) == 110
    @test maximum(local_array_fields) == 40
    @test minimum(local_array_fields) == 1
    @test all(x -> x > 0, local_array_fields)
    @test any(x -> x == 40, local_array_fields)

    threaded_u = ThreadedHaloArray(Int, (3,), 1; dims=(2,), boundary_condition=:repeating)
    threaded_v = similar(threaded_u)
    interior_view(threaded_u, 1) .= [1, 2, 3]
    interior_view(threaded_u, 2) .= [4, 5, 6]
    interior_view(threaded_v, 1) .= [10, 20, 30]
    interior_view(threaded_v, 2) .= [40, 50, 60]

    @test mapreduce(identity, +, threaded_u) == 21
    @test @inferred(mapreduce(identity, +, threaded_u)) == 21
    @test reduce(+, threaded_u) == 21
    @test sum(threaded_u) == 21
    @test sum(abs2, threaded_u) == 91
    @test maximum(threaded_u) == 6
    @test minimum(threaded_u) == 1
    @test mapfoldl(abs2, +, threaded_u) == 91
    @test mapfoldr(abs2, +, threaded_u) == 91
    @test mapreduce((x, y) -> x * y, +, threaded_u, threaded_v) == 910
    @test @inferred(mapreduce((x, y) -> x * y, +, threaded_u, threaded_v)) == 910
    @test mapreduce(x -> x[1] * x[2], +, zip(threaded_u, threaded_v)) == 910
    @test all(x -> x > 0, threaded_u)
    @test any(x -> x == 6, threaded_u)

    maybe_threaded = MaybeHaloArray(threaded_u)
    @test sum(maybe_threaded) == 21
    @test maximum(maybe_threaded) == 6
    @test minimum(maybe_threaded) == 1

    threaded_fields = MultiHaloArray((; u=threaded_u, v=threaded_v))
    @test mapreduce(identity, +, threaded_fields) == 231
    @test @inferred(mapreduce(identity, +, threaded_fields)) == 231
    @test sum(threaded_fields) == 231
    @test maximum(threaded_fields) == 60
    @test minimum(threaded_fields) == 1
    @test mapfoldl(identity, +, threaded_fields) == 231
    @test mapfoldr(identity, +, threaded_fields) == 231
    @test all(x -> x > 0, threaded_fields)
    @test any(x -> x == 60, threaded_fields)

    threaded_array_fields = ArrayOfHaloArray([threaded_u, threaded_v])
    @test mapreduce(identity, +, threaded_array_fields) == 231
    @test sum(threaded_array_fields) == 231
    @test maximum(threaded_array_fields) == 60
    @test minimum(threaded_array_fields) == 1
    @test all(x -> x > 0, threaded_array_fields)
    @test any(x -> x == 60, threaded_array_fields)

    @test mapreduce(identity, +, threaded_u; dims=1)[] == 21

    @testset "_combine_threaded_reduction" begin
        # 3-arg: prepends an initial result (used when tile 1 is computed separately)
        @test HaloArrays._combine_threaded_reduction(+, 100, [1, 2, 3]) == 106
        @test HaloArrays._combine_threaded_reduction(*, 2, [3, 4]) == 24
        @test HaloArrays._combine_threaded_reduction(max, 0, [3, 1, 4]) == 4

        # empty tail: returns initial result unchanged
        @test HaloArrays._combine_threaded_reduction(+, 7, Int[]) == 7
    end
end
