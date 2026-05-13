using Test
using HaloArrays

@testset "LocalHaloArray" begin
    @testset "1D boundary conditions" begin
        ha = LocalHaloArray(Int, (4,), 2; boundary_condition=((Repeating(), Reflecting()),))

        fill!(parent(ha), -1)
        interior_view(ha) .= [10, 20, 30, 40]

        boundary_condition!(ha)

        @test ha isa LocalHaloArray
        @test size(ha) == (4,)
        @test interior_size(ha) == (4,)
        @test full_size(ha) == (8,)
        @test halo_width(ha) == 2
        @test axes(ha) == (Base.OneTo(4),)
        @test collect(eachindex(ha)) == collect(eachindex(interior_view(ha)))
        @test collect(ha) == [10, 20, 30, 40]
        ha[2] = 25
        @test interior_view(ha)[2] == 25
        interior_view(ha)[2] = 20
        @test parent(ha)[1:2] == [10, 10]
        @test parent(ha)[3:6] == [10, 20, 30, 40]
        @test parent(ha)[7:8] == [40, 30]
        @test get_comm(ha) === nothing
        @test global_size(ha) == size(ha)
    end

    @testset "periodic boundaries wrap local interior" begin
        ha = LocalHaloArray(Int, (4,), 2; boundary_condition=:periodic)
        interior_view(ha) .= [10, 20, 30, 40]

        boundary_condition!(ha)

        @test parent(ha)[1:2] == [30, 40]
        @test parent(ha)[7:8] == [10, 20]
    end

    @testset "periodic flux contributions fold local ghosts" begin
        ha = LocalHaloArray(Int, (4,), 1; boundary_condition=:periodic)
        fill!(parent(ha), 0)
        interior_view(ha) .= [1, 2, 3, 4]
        parent(ha)[1] = 10
        parent(ha)[end] = 20

        synchronize_flux_contributions!(ha)

        @test collect(interior_view(ha)) == [21, 2, 3, 14]
        @test parent(ha)[1] == 10
        @test parent(ha)[end] == 20
    end

    @testset "non-periodic flux contributions leave local ghosts unchanged" begin
        ha = LocalHaloArray(Int, (4,), 1; boundary_condition=:repeating)
        fill!(parent(ha), 0)
        interior_view(ha) .= [1, 2, 3, 4]
        parent(ha)[1] = 10
        parent(ha)[end] = 20

        synchronize_flux_contributions!(ha)

        @test collect(interior_view(ha)) == [1, 2, 3, 4]
        @test parent(ha)[1] == 10
        @test parent(ha)[end] == 20
    end

    @testset "2D mixed boundaries" begin
        ha = LocalHaloArray(
            Int,
            (3, 4),
            1;
            boundary_condition=((Reflecting(), Repeating()), (Antireflecting(), Reflecting())),
        )

        fill!(parent(ha), -1)
        interior = interior_view(ha)
        for i in 1:size(ha, 1), j in 1:size(ha, 2)
            interior[i, j] = 10 * i + j
        end

        boundary_condition!(ha)

        @test collect(get_recv_view(Side(1), Dim(1), ha)) == reshape([11, 12, 13, 14], 1, 4)
        @test collect(get_recv_view(Side(2), Dim(1), ha)) == reshape([31, 32, 33, 34], 1, 4)
        @test collect(get_recv_view(Side(1), Dim(2), ha)) == reshape([-11, -21, -31], 3, 1)
        @test collect(get_recv_view(Side(2), Dim(2), ha)) == reshape([14, 24, 34], 3, 1)
    end

    @testset "similar, copy, and broadcast" begin
        ha = LocalHaloArray(Float64, (3,), 1; boundary_condition=:repeating)
        interior_view(ha) .= [1.0, 2.0, 3.0]

        shifted = ha .+ 2
        @test shifted isa LocalHaloArray
        @test collect(interior_view(shifted)) == [3.0, 4.0, 5.0]
        @test halo_width(shifted) == halo_width(ha)

        dest = similar(ha)
        dest .= 2 .* ha
        @test collect(interior_view(dest)) == [2.0, 4.0, 6.0]

        resized = similar(ha, Float32, (2,))
        @test eltype(resized) === Float32
        @test size(resized) == (2,)
        @test full_size(resized) == (4,)

        copied = copy(ha)
        interior_view(copied)[1] = -1
        @test interior_view(ha)[1] == 1.0
    end
end

@testset "LocalMultiHaloArray" begin
    u = LocalHaloArray(Float64, (3, 2), 1; boundary_condition=:repeating)
    v = LocalHaloArray(Int, (3, 2), 1; boundary_condition=:repeating)

    for i in 1:size(u, 1), j in 1:size(u, 2)
        interior_view(u)[i, j] = i + j / 10
        interior_view(v)[i, j] = 10 * i + j
    end

    fields = LocalMultiHaloArray((; u, v); check=true)

    @test fields isa LocalMultiHaloArray
    @test eltype(fields) === Float64
    @test ndims(fields) == 2
    @test fields[:u] === u
    @test fields[:v] === v
    @test isactive(fields)

    views = interior_view(fields)
    @test keys(views) == (:u, :v)
    @test collect(views.u) == [i + j / 10 for i in 1:3, j in 1:2]
    @test collect(views.v) == [10 * i + j for i in 1:3, j in 1:2]

    shifted = fields .+ 2
    @test shifted isa LocalMultiHaloArray
    @test collect(interior_view(shifted.arrays.u)) == [i + j / 10 + 2 for i in 1:3, j in 1:2]
    @test collect(interior_view(shifted.arrays.v)) == [10 * i + j + 2 for i in 1:3, j in 1:2]

    dest = similar(fields)
    dest .= 2 .* fields
    @test collect(interior_view(dest.arrays.u)) == [2 * (i + j / 10) for i in 1:3, j in 1:2]
    @test collect(interior_view(dest.arrays.v)) == [2 * (10 * i + j) for i in 1:3, j in 1:2]

    boundary_condition!(fields)
    @test parent(fields.arrays.u)[1, 2] == interior_view(u)[1, 1]

    bad = LocalHaloArray(Float64, (4, 2), 1; boundary_condition=:repeating)
    @test_throws DimensionMismatch LocalMultiHaloArray((; u, bad); check=true)
end
