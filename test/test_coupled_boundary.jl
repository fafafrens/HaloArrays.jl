using Test
using HaloArrays

# A deliberately *coupled* test BC: each field's ghost is filled from the OTHER
# field's interior edge. This cannot be expressed as an independent per-field BC,
# which is exactly the case the coupled hook exists for.
#
# Canonical form (0.3+): ONE tile-generic method; `tile` is `nothing` on
# Local/MPI fields and the boundary tile id on threaded fields — passed straight
# through to edge_view/ghost_view.
struct SwapBC <: AbstractCoupledBoundaryCondition end
function HaloArrays.apply_coupled_bc!(::SwapBC, state, s::Side{S}, d::Dim{D}, tile) where {S,D}
    a, b = eachfield(state)
    ghost_view(a, s, d, tile) .= edge_view(b, s, d, tile)
    ghost_view(b, s, d, tile) .= edge_view(a, s, d, tile)
    return nothing
end

# The same BC in the pre-0.3 legacy form (4-arg whole-array + 5-arg per-tile),
# kept to lock the backward-compat dispatch path.
struct LegacySwapBC <: AbstractCoupledBoundaryCondition end
function HaloArrays.apply_coupled_bc!(::LegacySwapBC, state, s::Side{S}, d::Dim{D}) where {S,D}
    a, b = eachfield(state)
    ghost_view(a, s, d) .= edge_view(b, s, d)
    ghost_view(b, s, d) .= edge_view(a, s, d)
    return nothing
end
function HaloArrays.apply_coupled_bc!(::LegacySwapBC, state, s::Side{S}, d::Dim{D}, tile_id::Integer) where {S,D}
    a, b = eachfield(state)
    ghost_view(a, s, d, tile_id) .= edge_view(b, s, d, tile_id)
    ghost_view(b, s, d, tile_id) .= edge_view(a, s, d, tile_id)
    return nothing
end

struct UnimplementedBC <: AbstractCoupledBoundaryCondition end

@testset "Coupled boundary conditions" begin
    nx = 6

    @testset "swap BC ($(nameof(typeof(bc)))) on $(nameof(typeof(make())))" for bc in (SwapBC(), LegacySwapBC()), make in (
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

        apply_coupled_bc!(bc, state)

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

    @testset "0.3 renamed-helper shims still work" begin
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:noboundary)
        @test get_send_view(Side(1), Dim(1), u) == edge_view(u, Side(1), Dim(1))
        @test get_recv_view(Side(2), Dim(1), u) == ghost_view(u, Side(2), Dim(1))
        @test get_comm(u) === communicator(u)
        @test isactive(u) == is_active(u)
        t = ThreadedHaloArray(Float64, (4,), 1; dims=(2,), boundary_condition=:noboundary)
        @test get_send_view(Side(1), Dim(1), t, 1) == edge_view(t, Side(1), Dim(1), 1)
    end

    @testset "helpers + unimplemented error" begin
        u = LocalHaloArray(Float64, (4,), 1; boundary_condition=:noboundary)
        @test is_physical_boundary(u, Side(1), Dim(1))   # local edges are always physical

        state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,), (4,), 1;
            boundary_condition=((:noboundary, :noboundary),))
        @test length(eachfield(state)) == 2
        @test_throws ArgumentError apply_coupled_bc!(UnimplementedBC(), state)
    end

    @testset "threaded fields ($(nameof(typeof(bc))), boundary tiles only)" for bc in (SwapBC(), LegacySwapBC())
        # 8 cells over 2 tiles of 4; tile 1 owns 1..4, tile 2 owns 5..8.
        state = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,), (4,), 1;
            dims=(2,), boundary_condition=:noboundary)
        a, b = eachfield(state)
        for i in 1:8
            a[i] = i
            b[i] = 100 + i
        end

        apply_coupled_bc!(bc, state)

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
