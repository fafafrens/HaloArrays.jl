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
    nx, ny = local_size(ha)
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
    nx, ny, nz = local_size(ha)
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

function _check_nonperiodic_1d_synchronize_halo_boundary_conditions()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(false,))
    halo_width_cells = 2
    local_cells = 5
    ha = HaloArray(
        Int,
        (local_cells,),
        halo_width_cells,
        topology;
        boundary_condition=((Reflecting(), Antireflecting()),),
    )

    _fill_rank_pattern!(ha, rank)
    MPI.Barrier(comm)
    synchronize_halo!(ha)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]

    expected_left = if left_rank == MPI.PROC_NULL
        [100 * rank + 2, 100 * rank + 1]
    else
        [100 * left_rank + local_cells - halo_width_cells + i for i in 1:halo_width_cells]
    end
    expected_right = if right_rank == MPI.PROC_NULL
        [-(100 * rank + local_cells), -(100 * rank + local_cells - 1)]
    else
        [100 * right_rank + i for i in 1:halo_width_cells]
    end

    @test collect(parent(ha)[1:halo_width_cells]) == expected_left
    @test collect(parent(ha)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)]) ==
        expected_right
    @test collect(interior_view(ha)) == [100 * rank + i for i in 1:local_cells]

    return nothing
end

function _check_multihaloarray_public_split_exchange_with_boundaries()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(false,))
    local_cells = 4
    u = HaloArray(
        Int,
        (local_cells,),
        1,
        topology;
        boundary_condition=((Repeating(), Reflecting()),),
    )
    v = HaloArray(
        Int,
        (local_cells,),
        1,
        topology;
        boundary_condition=((Antireflecting(), Repeating()),),
    )
    fields = MultiHaloArray((; u, v))

    for i in 1:local_cells
        interior_view(u)[i] = 100 * rank + i
        interior_view(v)[i] = 1000 * rank + 10 * i
    end

    MPI.Barrier(comm)
    @test start_halo_exchange!(fields) === fields
    @test finish_halo_exchange!(fields) === fields
    boundary_condition!(fields)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]

    expected_u_left = left_rank == MPI.PROC_NULL ? 100 * rank + 1 : 100 * left_rank + local_cells
    expected_u_right = right_rank == MPI.PROC_NULL ? 100 * rank + local_cells : 100 * right_rank + 1
    expected_v_left = left_rank == MPI.PROC_NULL ? -(1000 * rank + 10) : 1000 * left_rank + 10 * local_cells
    expected_v_right = right_rank == MPI.PROC_NULL ? 1000 * rank + 10 * local_cells : 1000 * right_rank + 10

    @test parent(u)[1] == expected_u_left
    @test parent(u)[end] == expected_u_right
    @test parent(v)[1] == expected_v_left
    @test parent(v)[end] == expected_v_right
    @test collect(interior_view(u)) == [100 * rank + i for i in 1:local_cells]
    @test collect(interior_view(v)) == [1000 * rank + 10 * i for i in 1:local_cells]

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

    fields = MultiHaloArray((; u, v))
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

function _check_arrayofhaloarray_broadcast()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    arrays = [HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))
              for _ in 1:2, _ in 1:2]

    for I in CartesianIndices(arrays)
        interior = interior_view(arrays[I])
        for i in eachindex(interior)
            interior[i] = 100 * I[1] + 10 * I[2] + rank + i
        end
    end

    fields = ArrayOfHaloArray(arrays)
    shifted = fields .+ 3
    @test shifted isa ArrayOfHaloArray

    for I in CartesianIndices(arrays)
        expected = [100 * I[1] + 10 * I[2] + rank + i + 3 for i in 1:4]
        @test collect(interior_view(shifted[I])) == expected
    end

    dest = similar(fields)
    dest .= 2 .* fields .- shifted

    for I in CartesianIndices(arrays)
        expected = [100 * I[1] + 10 * I[2] + rank + i - 3 for i in 1:4]
        @test collect(interior_view(dest[I])) == expected
    end

    MPI.Barrier(comm)
    halo_exchange!(dest)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    for I in CartesianIndices(arrays)
        @test parent(dest[I])[1] == 100 * I[1] + 10 * I[2] + left_rank + 4 - 3
        @test parent(dest[I])[end] == 100 * I[1] + 10 * I[2] + right_rank + 1 - 3
    end

    return nothing
end

function _check_nested_multihaloarray_broadcast()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    rho = HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))
    q_arrays = [HaloArray(Float64, (4,), 1, topology; boundary_condition=((Periodic(), Periodic()),))
                for _ in 1:2]

    for i in eachindex(interior_view(rho))
        interior_view(rho)[i] = rank + i
    end

    for c in eachindex(q_arrays)
        interior = interior_view(q_arrays[c])
        for i in eachindex(interior)
            interior[i] = 100 * c + rank + i
        end
    end

    fields = MultiHaloArray((; rho, q=ArrayOfHaloArray(q_arrays)))
    shifted = fields .+ 4
    @test shifted isa MultiHaloArray
    @test shifted.arrays.q isa ArrayOfHaloArray
    @test collect(interior_view(shifted.arrays.rho)) == [rank + i + 4 for i in 1:4]
    @test collect(interior_view(shifted.arrays.q[1])) == [100 + rank + i + 4 for i in 1:4]
    @test collect(interior_view(shifted.arrays.q[2])) == [200 + rank + i + 4 for i in 1:4]

    dest = similar(fields)
    dest .= fields .+ shifted

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]

    MPI.Barrier(comm)
    halo_exchange!(dest)
    MPI.Barrier(comm)

    @test parent(dest.arrays.rho)[1] == 2 * (left_rank + 4) + 4
    @test parent(dest.arrays.rho)[end] == 2 * (right_rank + 1) + 4
    for c in eachindex(dest.arrays.q.arrays)
        @test parent(dest.arrays.q[c])[1] == 2 * (100 * c + left_rank + 4) + 4
        @test parent(dest.arrays.q[c])[end] == 2 * (100 * c + right_rank + 1) + 4
    end

    return nothing
end

function _check_periodic_1d_flux_contribution_exchange()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic = (true,))
    halo_width_cells = 2
    local_cells = 6
    du = HaloArray(
        Int,
        (local_cells,),
        halo_width_cells,
        topology;
        boundary_condition = ((Periodic(), Periodic()),),
    )

    fill!(parent(du), 0)
    interior = interior_view(du)
    for i in 1:local_cells
        interior[i] = 10 * rank + i
    end

    left_ghost_values = [1000 * rank + 10 + i for i in 1:halo_width_cells]
    right_ghost_values = [1000 * rank + 20 + i for i in 1:halo_width_cells]
    parent(du)[1:halo_width_cells] .= left_ghost_values
    parent(du)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)] .= right_ghost_values

    MPI.Barrier(comm)
    synchronize_flux_contributions!(du)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    expected = [10 * rank + i for i in 1:local_cells]
    expected[1:halo_width_cells] .+= [1000 * left_rank + 20 + i for i in 1:halo_width_cells]
    expected[(local_cells - halo_width_cells + 1):local_cells] .+=
        [1000 * right_rank + 10 + i for i in 1:halo_width_cells]

    @test collect(interior_view(du)) == expected
    @test collect(parent(du)[1:halo_width_cells]) == left_ghost_values
    @test collect(parent(du)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)]) ==
        right_ghost_values

    return nothing
end

function _check_nonperiodic_1d_flux_contribution_exchange()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(false,))
    halo_width_cells = 2
    local_cells = 6
    du = HaloArray(
        Int,
        (local_cells,),
        halo_width_cells,
        topology;
        boundary_condition=((Repeating(), Repeating()),),
    )

    fill!(parent(du), 0)
    interior = interior_view(du)
    for i in 1:local_cells
        interior[i] = 10 * rank + i
    end

    left_ghost_values = [1000 * rank + 10 + i for i in 1:halo_width_cells]
    right_ghost_values = [1000 * rank + 20 + i for i in 1:halo_width_cells]
    parent(du)[1:halo_width_cells] .= left_ghost_values
    parent(du)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)] .= right_ghost_values

    MPI.Barrier(comm)
    synchronize_flux_contributions!(du)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    expected = [10 * rank + i for i in 1:local_cells]
    if left_rank != MPI.PROC_NULL
        expected[1:halo_width_cells] .+= [1000 * left_rank + 20 + i for i in 1:halo_width_cells]
    end
    if right_rank != MPI.PROC_NULL
        expected[(local_cells - halo_width_cells + 1):local_cells] .+=
            [1000 * right_rank + 10 + i for i in 1:halo_width_cells]
    end

    @test collect(interior_view(du)) == expected
    @test collect(parent(du)[1:halo_width_cells]) == left_ghost_values
    @test collect(parent(du)[(halo_width_cells + local_cells + 1):(2 * halo_width_cells + local_cells)]) ==
        right_ghost_values

    return nothing
end

function _ghost_face_value(rank, dim, side, I...)
    return 100_000 * rank + 10_000 * dim + 1_000 * side + sum(10^(length(I) - i) * I[i] for i in eachindex(I))
end

function _check_periodic_2d_flux_contribution_exchange()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0); periodic = (true, true))
    local_size = (4, 5)
    nx, ny = local_size
    du = HaloArray(
        Int,
        local_size,
        1,
        topology;
        boundary_condition = ((Periodic(), Periodic()), (Periodic(), Periodic())),
    )

    fill!(parent(du), 0)
    data = parent(du)
    for j in 1:ny
        data[1, j + 1] = _ghost_face_value(rank, 1, 1, j)
        data[nx + 2, j + 1] = _ghost_face_value(rank, 1, 2, j)
    end
    for i in 1:nx
        data[i + 1, 1] = _ghost_face_value(rank, 2, 1, i)
        data[i + 1, ny + 2] = _ghost_face_value(rank, 2, 2, i)
    end

    MPI.Barrier(comm)
    synchronize_flux_contributions!(du)
    MPI.Barrier(comm)

    expected = zeros(Int, local_size)
    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    down_rank = topology.neighbors[2][1]
    up_rank = topology.neighbors[2][2]

    for j in 1:ny
        expected[1, j] += _ghost_face_value(left_rank, 1, 2, j)
        expected[nx, j] += _ghost_face_value(right_rank, 1, 1, j)
    end
    for i in 1:nx
        expected[i, 1] += _ghost_face_value(down_rank, 2, 2, i)
        expected[i, ny] += _ghost_face_value(up_rank, 2, 1, i)
    end

    @test collect(interior_view(du)) == expected

    return nothing
end

function _check_nonperiodic_2d_flux_contribution_exchange()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0, 0); periodic=(false, false))
    local_size = (4, 5)
    nx, ny = local_size
    du = HaloArray(
        Int,
        local_size,
        1,
        topology;
        boundary_condition=((Repeating(), Repeating()), (Repeating(), Repeating())),
    )

    fill!(parent(du), 0)
    data = parent(du)
    for j in 1:ny
        data[1, j + 1] = _ghost_face_value(rank, 1, 1, j)
        data[nx + 2, j + 1] = _ghost_face_value(rank, 1, 2, j)
    end
    for i in 1:nx
        data[i + 1, 1] = _ghost_face_value(rank, 2, 1, i)
        data[i + 1, ny + 2] = _ghost_face_value(rank, 2, 2, i)
    end

    MPI.Barrier(comm)
    synchronize_flux_contributions!(du)
    MPI.Barrier(comm)

    expected = zeros(Int, local_size)
    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]
    down_rank = topology.neighbors[2][1]
    up_rank = topology.neighbors[2][2]

    if left_rank != MPI.PROC_NULL
        for j in 1:ny
            expected[1, j] += _ghost_face_value(left_rank, 1, 2, j)
        end
    end
    if right_rank != MPI.PROC_NULL
        for j in 1:ny
            expected[nx, j] += _ghost_face_value(right_rank, 1, 1, j)
        end
    end
    if down_rank != MPI.PROC_NULL
        for i in 1:nx
            expected[i, 1] += _ghost_face_value(down_rank, 2, 2, i)
        end
    end
    if up_rank != MPI.PROC_NULL
        for i in 1:nx
            expected[i, ny] += _ghost_face_value(up_rank, 2, 1, i)
        end
    end

    @test collect(interior_view(du)) == expected

    return nothing
end

function _check_multihaloarray_flux_contribution_exchange()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic = (true,))
    u = HaloArray(Int, (4,), 1, topology; boundary_condition = ((Periodic(), Periodic()),))
    v = HaloArray(Int, (4,), 1, topology; boundary_condition = ((Periodic(), Periodic()),))
    fields = MultiHaloArray((; u, v))

    fill!(parent(u), 0)
    fill!(parent(v), 0)
    parent(u)[1] = 10 * rank + 1
    parent(u)[end] = 10 * rank + 2
    parent(v)[1] = 100 * rank + 1
    parent(v)[end] = 100 * rank + 2

    MPI.Barrier(comm)
    synchronize_flux_contributions!(fields)
    MPI.Barrier(comm)

    left_rank = topology.neighbors[1][1]
    right_rank = topology.neighbors[1][2]

    @test collect(interior_view(u)) == [10 * left_rank + 2, 0, 0, 10 * right_rank + 1]
    @test collect(interior_view(v)) == [100 * left_rank + 2, 0, 0, 100 * right_rank + 1]

    return nothing
end

function _check_haloarray_scalar_indexing()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    @test nranks > 1

    topology = CartesianTopology(comm, (0,); periodic=(true,))
    local_cells = 4
    ha = HaloArray(Int, (local_cells,), 1, topology; boundary_condition=:periodic)
    _fill_rank_pattern!(ha, rank)

    coord = topology.cart_coords[1]
    owned_global = coord * local_cells + 2
    remote_coord = mod(coord + 1, topology.dims[1])
    remote_global = remote_coord * local_cells + 1

    @test ha[owned_global] == 100 * rank + 2
    setindex!(ha, -rank - 1, owned_global)
    @test interior_view(ha)[2] == -rank - 1
    @test_throws ArgumentError ha[remote_global]
    @test_throws BoundsError ha[0]

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

@testset "MPI synchronize_halo! with physical boundaries" begin
    _check_nonperiodic_1d_synchronize_halo_boundary_conditions()
    _check_multihaloarray_public_split_exchange_with_boundaries()
end

@testset "MPI broadcast over halo arrays" begin
    _check_haloarray_broadcast()
    _check_multihaloarray_broadcast()
    _check_arrayofhaloarray_broadcast()
    _check_nested_multihaloarray_broadcast()
end

@testset "MPI HaloArray scalar indexing" begin
    _check_haloarray_scalar_indexing()
end

@testset "MPI flux contribution exchange across ranks" begin
    _check_periodic_1d_flux_contribution_exchange()
    _check_nonperiodic_1d_flux_contribution_exchange()
    _check_periodic_2d_flux_contribution_exchange()
    _check_nonperiodic_2d_flux_contribution_exchange()
    _check_multihaloarray_flux_contribution_exchange()
end
