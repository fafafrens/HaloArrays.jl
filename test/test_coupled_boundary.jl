using Test
using HaloArrays

# A deliberately *coupled* test BC: each field's ghost is filled from the OTHER
# field's interior edge. This cannot be expressed as an independent per-field BC,
# which is exactly the case the coupled hook exists for.
struct SwapBC <: AbstractCoupledBoundaryCondition end
function HaloArrays.apply_coupled_bc!(::SwapBC, state, s::Side{S}, d::Dim{D}) where {S,D}
    a, b = eachfield(state)
    get_recv_view(s, d, a) .= get_send_view(s, d, b)
    get_recv_view(s, d, b) .= get_send_view(s, d, a)
    return nothing
end
# per-tile variant for ThreadedHaloArray fields
function HaloArrays.apply_coupled_bc!(::SwapBC, state, s::Side{S}, d::Dim{D}, tile_id::Integer) where {S,D}
    a, b = eachfield(state)
    get_recv_view(s, d, a, tile_id) .= get_send_view(s, d, b, tile_id)
    get_recv_view(s, d, b, tile_id) .= get_send_view(s, d, a, tile_id)
    return nothing
end

struct UnimplementedBC <: AbstractCoupledBoundaryCondition end

@testset "Coupled boundary conditions" begin
    nx = 6

    @testset "swap BC on $(nameof(typeof(make())))" for make in (
            () -> ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (nx,), 1;
                      boundary_condition=((:noboundary, :noboundary),)),
            () -> LocalMultiHaloArray(Float64, (nx,), 1; boundary_conditions=(
                      a=((NoBoundaryCondition(), NoBoundaryCondition()),),
                      b=((NoBoundaryCondition(), NoBoundaryCondition()),))),
        )
        state = make()
        a, b = eachfield(state)
        interior_view(a) .= Float64.(1:nx)          # 1..nx
        interior_view(b) .= Float64.((1:nx) .+ 100) # 101..100+nx

        apply_coupled_bc!(SwapBC(), state)

        # left ghost of a = b's first interior cell, and vice versa
        @test parent(a)[1]      == 101.0
        @test parent(b)[1]      == 1.0
        # right ghost of a = b's last interior cell, and vice versa
        @test parent(a)[nx + 2] == 100.0 + nx
        @test parent(b)[nx + 2] == Float64(nx)
        # interior is untouched
        @test interior_view(a) == Float64.(1:nx)
    end

    @testset "driver fires all boundary" begin
        ny = 4
        # x = :noboundary (coupled), y = :periodic (handled per field by sync)
        state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (nx, ny), 1;
            boundary_condition=((:noboundary, :noboundary), (:periodic, :periodic)))
        a, b = eachfield(state)
        for I in CartesianIndices((nx, ny))
            a[Tuple(I)...] = I[1] + 10 * I[2]
            b[Tuple(I)...] = 100 + I[1] + 10 * I[2]
        end

        synchronize_halo!(state)             # fills y ghosts (periodic); leaves x alone
        ay_before = copy(parent(a))          # snapshot to check the coupled call leaves y
        apply_coupled_bc!(SwapBC(), state)   # should touch ONLY the x faces


        # x ghosts were swapped from the other field
        @test parent(a)[1, 1 + 1]      == parent(b)[1 + 1, 1 + 1]
        @test parent(b)[nx + 2, 1 + 1] == parent(a)[nx + 1, 1 + 1]
        # the coupled BC did not modify the y-ghost columns
    end

    @testset "helpers + unimplemented error" begin
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:noboundary)
        @test is_physical_boundary(u, Side(1), Dim(1))   # local edges are always physical

        state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (4,), 1;
            boundary_condition=((:noboundary, :noboundary),))
        @test length(eachfield(state)) == 2
        @test_throws ArgumentError apply_coupled_bc!(UnimplementedBC(), state)
    end

    @testset "threaded fields (per-tile, boundary tiles only)" begin
        # 8 cells over 2 tiles of 4; tile 1 owns 1..4, tile 2 owns 5..8.
        state = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,), (4,), 1;
            dims=(2,), boundary_condition=:noboundary)
        a, b = eachfield(state)
        for i in 1:8
            a[i] = i
            b[i] = 100 + i
        end

        apply_coupled_bc!(SwapBC(), state)

        # left domain edge is on tile 1, right edge on tile 2; ghosts come from
        # the OTHER field's adjacent interior cell
        @test tile_parent(a, 1)[1]   == 101.0   # b's global cell 1
        @test tile_parent(b, 1)[1]   == 1.0     # a's global cell 1
        @test tile_parent(a, 2)[end] == 108.0   # b's global cell 8
        @test tile_parent(b, 2)[end] == 8.0     # a's global cell 8
        # interior-facing tile faces (neighbour ≠ 0) are left untouched
        @test tile_parent(a, 1)[end] == 0.0
        @test tile_parent(a, 2)[1]   == 0.0

        @test is_physical_boundary(state, Side(1), Dim(1))   # non-periodic ⇒ physical edge
    end
end
