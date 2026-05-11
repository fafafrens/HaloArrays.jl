using Test
using MPI
using HaloArrays

const EXCHANGE_IMPLEMENTATIONS = (
    blocking=halo_exchange!,
    waitall=HaloArrays.halo_exchange_waitall!,
    waitall_unsafe=HaloArrays.halo_exchange_waitall_unsafe!,
    async=HaloArrays.halo_exchange_async!,
    async_unsafe=HaloArrays.halo_exchange_async_unsafe!,
    split_async=(ha -> begin
        HaloArrays.start_halo_exchange_async!(ha)
        HaloArrays.end_halo_exchange_wait!(ha)
    end),
    split_async_unsafe=(ha -> begin
        HaloArrays.start_halo_exchange_async_unsafe!(ha)
        HaloArrays.end_halo_exchange_async_wait_unsafe!(ha)
    end),
    public_split=(ha -> begin
        start_halo_exchange!(ha)
        finish_halo_exchange!(ha)
    end),
)

function _fill_rank_pattern!(ha, rank)
    interior = interior_view(ha)
    for i in eachindex(interior)
        interior[i] = 100 * rank + i
    end
    return ha
end

function _check_periodic_1d_halo_exchange!(exchange!)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    halo_width_cells = 2
    local_cells = 5
    ha = HaloArray(
        Int,
        (local_cells,),
        halo_width_cells,
        topology;
        boundary_condition=((Periodic(), Periodic()),),
    )

    _fill_rank_pattern!(ha, rank)
    MPI.Barrier(comm)
    exchange!(ha)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    @test left_rank != MPI.PROC_NULL
    @test right_rank != MPI.PROC_NULL

    left_halo = collect(parent(ha)[1:halo_width_cells])
    right_halo = collect(parent(ha)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)])
    interior = collect(interior_view(ha))

    expected_left = [100 * left_rank + local_cells - halo_width_cells + i for i in 1:halo_width_cells]
    expected_right = [100 * right_rank + i for i in 1:halo_width_cells]
    expected_interior = [100 * rank + i for i in 1:local_cells]

    @test left_halo == expected_left
    @test right_halo == expected_right
    @test interior == expected_interior

    return nothing
end

function _fill_2d_rank_pattern!(ha, rank)
    nx, ny = size(ha)
    interior = interior_view(ha)
    for i in 1:nx, j in 1:ny
        interior[i, j] = 1000 * rank + 100 * i + j
    end
    return ha
end

function _check_periodic_2d_halo_exchange!(exchange!)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    halo_width_cells = 1
    local_size = (4, 3)
    ha = HaloArray(
        Int,
        local_size,
        halo_width_cells,
        topology;
        boundary_condition=((Periodic(), Periodic()), (Periodic(), Periodic())),
    )

    _fill_2d_rank_pattern!(ha, rank)
    MPI.Barrier(comm)
    exchange!(ha)
    MPI.Barrier(comm)

    nx, ny = local_size
    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    down_rank = topology.neighbors[2][1]
    up_rank = topology.neighbors[2][2]

    @test collect(parent(ha)[1, 2:(ny + 1)]) == [1000 * left_rank + 100 * nx + j for j in 1:ny]
    @test collect(parent(ha)[nx + 2, 2:(ny + 1)]) == [1000 * right_rank + 100 + j for j in 1:ny]
    @test collect(parent(ha)[2:(nx + 1), 1]) == [1000 * down_rank + 100 * i + ny for i in 1:nx]
    @test collect(parent(ha)[2:(nx + 1), ny + 2]) == [1000 * up_rank + 100 * i + 1 for i in 1:nx]
    @test collect(interior_view(ha)) == [1000 * rank + 100 * i + j for i in 1:nx, j in 1:ny]

    return nothing
end

function _fill_3d_rank_pattern!(ha, rank)
    nx, ny, nz = size(ha)
    interior = interior_view(ha)
    for i in 1:nx, j in 1:ny, k in 1:nz
        interior[i, j, k] = 100_000 * rank + 10_000 * i + 100 * j + k
    end
    return ha
end

function _value_3d(rank, i, j, k)
    return 100_000 * rank + 10_000 * i + 100 * j + k
end

function _check_periodic_3d_halo_exchange!(exchange!)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0, 0); periodic=(true, true, true))
    halo_width_cells = 1
    local_size = (3, 2, 2)
    ha = HaloArray(
        Int,
        local_size,
        halo_width_cells,
        topology;
        boundary_condition=ntuple(_ -> (Periodic(), Periodic()), 3),
    )

    _fill_3d_rank_pattern!(ha, MPI.Comm_rank(comm))
    MPI.Barrier(comm)
    exchange!(ha)
    MPI.Barrier(comm)

    nx, ny, nz = local_size
    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    down_rank = topology.neighbors[2][1]
    up_rank = topology.neighbors[2][2]
    back_rank = topology.neighbors[3][1]
    front_rank = topology.neighbors[3][2]

    @test collect(parent(ha)[1, 2:(ny + 1), 2:(nz + 1)]) ==
        [_value_3d(left_rank, nx, j, k) for j in 1:ny, k in 1:nz]
    @test collect(parent(ha)[nx + 2, 2:(ny + 1), 2:(nz + 1)]) ==
        [_value_3d(right_rank, 1, j, k) for j in 1:ny, k in 1:nz]
    @test collect(parent(ha)[2:(nx + 1), 1, 2:(nz + 1)]) ==
        [_value_3d(down_rank, i, ny, k) for i in 1:nx, k in 1:nz]
    @test collect(parent(ha)[2:(nx + 1), ny + 2, 2:(nz + 1)]) ==
        [_value_3d(up_rank, i, 1, k) for i in 1:nx, k in 1:nz]
    @test collect(parent(ha)[2:(nx + 1), 2:(ny + 1), 1]) ==
        [_value_3d(back_rank, i, j, nz) for i in 1:nx, j in 1:ny]
    @test collect(parent(ha)[2:(nx + 1), 2:(ny + 1), nz + 2]) ==
        [_value_3d(front_rank, i, j, 1) for i in 1:nx, j in 1:ny]

    return nothing
end

function _check_exchange_implementations_agree()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    topology = CartesianTopology(comm, (0, 0); periodic=(true, true))
    base = HaloArray(
        Int,
        (5, 4),
        1,
        topology;
        boundary_condition=((Periodic(), Periodic()), (Periodic(), Periodic())),
    )

    _fill_2d_rank_pattern!(base, rank)

    exchanged = map(values(EXCHANGE_IMPLEMENTATIONS)) do exchange!
        ha = copy(base)
        MPI.Barrier(comm)
        exchange!(ha)
        MPI.Barrier(comm)
        return ha
    end

    reference = collect(parent(first(exchanged)))
    for ha in exchanged
        @test collect(parent(ha)) == reference
    end

    return nothing
end

function _check_haloarray_broadcast()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    a = HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))
    b = similar(a)

    a_interior = interior_view(a)
    b_interior = interior_view(b)
    for i in eachindex(a_interior)
        a_interior[i] = rank + i
        b_interior[i] = 10 * rank + 2 * i
    end

    c = 2 .* a .+ b .+ 1
    @test c isa HaloArray
    @test collect(interior_view(c)) == [2 * (rank + i) + (10 * rank + 2 * i) + 1 for i in 1:4]

    c .= a .- b
    @test collect(interior_view(c)) == [(rank + i) - (10 * rank + 2 * i) for i in 1:4]

    MPI.Barrier(comm)
    halo_exchange!(c)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    @test parent(c)[1] == (left_rank + 4) - (10 * left_rank + 2 * 4)
    @test parent(c)[end] == (right_rank + 1) - (10 * right_rank + 2)

    return nothing
end

function _check_multihaloarray_broadcast()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    u = HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))
    v = HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))

    u_interior = interior_view(u)
    v_interior = interior_view(v)
    for i in eachindex(u_interior)
        u_interior[i] = rank + i
        v_interior[i] = 100 * rank + 10 * i
    end

    fields = MultiHaloArray((; u, v); check=true)
    shifted = fields .+ 3
    @test shifted isa MultiHaloArray
    @test collect(interior_view(shifted.arrays.u)) == [rank + i + 3 for i in 1:4]
    @test collect(interior_view(shifted.arrays.v)) == [100 * rank + 10 * i + 3 for i in 1:4]

    dest = similar(fields)
    dest .= 2 .* fields
    @test collect(interior_view(dest.arrays.u)) == [2 * (rank + i) for i in 1:4]
    @test collect(interior_view(dest.arrays.v)) == [2 * (100 * rank + 10 * i) for i in 1:4]

    return nothing
end

@testset "MPI halo exchange across ranks" begin
    for (name, exchange!) in pairs(EXCHANGE_IMPLEMENTATIONS)
        @testset "1D $name" begin
            _check_periodic_1d_halo_exchange!(exchange!)
        end
        @testset "2D $name" begin
            _check_periodic_2d_halo_exchange!(exchange!)
        end
        @testset "3D $name" begin
            _check_periodic_3d_halo_exchange!(exchange!)
        end
    end
    _check_exchange_implementations_agree()
end

@testset "MPI broadcast over halo arrays" begin
    _check_haloarray_broadcast()
    _check_multihaloarray_broadcast()
end
