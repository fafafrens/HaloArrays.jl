using Test
using MPI
using HaloArrays

function _test_topology(dims::NTuple{N,Int}) where {N}
    return CartesianTopology(MPI.COMM_SELF, dims; periodic=ntuple(_ -> false, Val(N)))
end

@testset "MultiHaloArray" begin
    topology = _test_topology((1, 1))
    u = HaloArray(Float64, (3, 2), 1, topology; boundary_condition=:repeating)
    v = HaloArray(Int, (3, 2), 1, topology; boundary_condition=:repeating)

    u_interior = interior_view(u)
    v_interior = interior_view(v)
    for i in 1:size(u, 1), j in 1:size(u, 2)
        u_interior[i, j] = i + j / 10
        v_interior[i, j] = 10 * i + j
    end

    fields = MultiHaloArray((; u, v); check=true)

    @test fields isa MultiHaloArray
    @test eltype(fields) === Float64
    @test ndims(fields) == 2
    @test HaloArrays.n_field(fields) == 2
    @test fields[:u] === u
    @test fields[:v] === v
    @test isactive(fields)

    views = interior_view(fields)
    @test keys(views) == (:u, :v)
    @test collect(views.u) == [i + j / 10 for i in 1:3, j in 1:2]
    @test collect(views.v) == [10 * i + j for i in 1:3, j in 1:2]

    shifted = fields .+ 2
    @test shifted isa MultiHaloArray
    @test collect(interior_view(shifted.arrays.u)) == [i + j / 10 + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(shifted.arrays.v)) == [10 * i + j + 2 for i in 1:3, j in 1:2]

    dest = similar(fields)
    dest .= 2 .* fields
    @test collect(interior_view(dest.arrays.u)) == [2 * (i + j / 10) for i in 1:3, j in 1:2]
    @test collect(interior_view(dest.arrays.v)) == [2 * (10 * i + j) for i in 1:3, j in 1:2]

    copied = copy(fields)
    interior_view(copied.arrays.u)[1, 1] = -1
    @test interior_view(fields.arrays.u)[1, 1] != interior_view(copied.arrays.u)[1, 1]

    bad = HaloArray(Float64, (4, 2), 1, topology; boundary_condition=:repeating)
    @test_throws DimensionMismatch MultiHaloArray((; u, bad); check=true)

    @test all(x -> x > 0, fields)
    @test any(x -> x == 22, fields)
end
