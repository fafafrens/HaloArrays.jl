using Test, HaloArrays, MPI
using StaticArrays: MVector

MPI.Initialized() || MPI.Init()

function exercise_field_access(state, tile=nothing)
    I = first(interior_cells(CellRanges(state)))
    v = [1.0, 2.0, 3.0, 4.0]
    out = zeros(4)
    @test scatter_fields!(state, I, v, tile) === state
    @test gather_fields!(out, state, I, tile) === out
    @test out == v
    @test add_fields!(state, I, v, -0.5, tile) === state
    gather_fields!(out, state, I, tile)
    @test out == v / 2
    @inbounds scatter_fields!(state, I, v, tile)
    @inbounds add_fields!(state, I, v, -0.5, tile)
    @inbounds gather_fields!(out, state, I, tile)
    @test out == v / 2
    m = MVector{4,Float64}(undef)
    gather_fields!(m, state, I, tile)
    @test m == out
    @test_throws DimensionMismatch gather_fields!(zeros(3), state, I, tile)
    @test_throws DimensionMismatch gather_fields!(out, state, CartesianIndex(1,1,1), tile)
    @test_throws BoundsError scatter_fields!(state, CartesianIndex(0,1), v, tile)
    @test_throws BoundsError add_fields!(state, I, v, 1, 0)
    gather_fields!(out, state, I, tile)
    @test out == v / 2
end

# Measure inside compiled functions, with preallocated buffers.
function access_allocations(v, state, I, tile)
    a = @allocated gather_fields!(v, state, I, tile)
    b = @allocated scatter_fields!(state, I, v, tile)
    c = @allocated add_fields!(state, I, v, 0.5, tile)
    return (a,b,c)
end

@testset "Field gather, scatter, and accumulation" begin
    local_state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,2), (3,2), 1;
                                   boundary_condition=:periodic)
    exercise_field_access(local_state)
    I = first(interior_cells(CellRanges(local_state)))
    scatter_fields!(local_state, I, [11.,12.,21.,22.])
    @test parent(local_state[1,1])[I] == 11
    @test parent(local_state[2,1])[I] == 12
    @test parent(local_state[1,2])[I] == 21
    @test parent(local_state[2,2])[I] == 22
    @inbounds scatter_fields!(local_state, I, [11.,12.,21.,22.])
    @inbounds add_fields!(local_state, I, zeros(4), 1.0)
    unchecked_out = zeros(4)
    @inbounds gather_fields!(unchecked_out, local_state, I)
    @test unchecked_out == [11,12,21,22]
    synchronize_halo!(local_state)
    out = zeros(4)
    gather_fields!(out, local_state, CartesianIndex(5,2))
    @test out == [11,12,21,22]

    named = LocalMultiHaloArray(Float64, (3,2), 1;
                                fields=(:a,:b,:c,:d), boundary_condition=:periodic)
    exercise_field_access(named)
    @test parent(named.b)[I] == 1

    threaded = ArrayOfHaloArray(ThreadedHaloArray, Float64, (4,), (3,2), 1;
                                dims=(2,1), boundary_condition=:periodic)
    exercise_field_access(threaded, 2)
    @test_throws ArgumentError gather_fields!(out, threaded, I)
    @test_throws BoundsError gather_fields!(out, threaded, I, 3)
    @test tile_parent(threaded[1],1)[I] == 0

    topology = CartesianTopology(MPI.COMM_SELF, (1,1); periodic=(true,true))
    distributed = ArrayOfHaloArray(HaloArray, Float64, (3,2), 1, topology;
                                   boundary_conditions=fill(:periodic,4))
    exercise_field_access(distributed)
    for (state,tile) in ((local_state,nothing),(named,nothing),(threaded,2),(distributed,nothing))
        access_allocations(out,state,I,tile)
        @test access_allocations(out,state,I,tile) == (0,0,0)
    end
end
