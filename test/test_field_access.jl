using Test, HaloArrays, MPI, LinearAlgebra
using StaticArrays: SVector
MPI.Initialized() || MPI.Init()

function site_allocations(state, I, tile, buffer)
    scalar = @allocated begin
        q = siteview(state, I, tile)
        q[1] = q[1] + 1
    end
    q = siteview(state, I, tile)
    gather = @allocated copyto!(buffer, q)
    scatter = @allocated copyto!(q, buffer)
    accumulation = @allocated q .+= 0.5 .* buffer
    return (scalar, gather, scatter, accumulation)
end

function exercise_siteview(state, tile=nothing)
    I = first(interior_cells(CellRanges(state)))
    q = @inferred siteview(state, I, tile)
    @test q isa AbstractVector{Float64}
    @test parent(q) === state
    @test size(q) == (4,)
    @test size(q, 2) == 1
    @test axes(q) == (Base.OneTo(4),)
    @test IndexStyle(typeof(q)) == IndexLinear()
    v = [1., 2., 3., 4.]
    @test copyto!(q, v) === q
    @test (@inferred q[1]) === 1.0
    @test collect(q) == v
    out = zeros(4)
    @test copyto!(out, q) === out
    @test out == v
    for T in (Float32, Float64)
        permuted = PermutedDimsArray(zeros(T, 4), (1,))
        @test copyto!(permuted, q) === permuted
        @test permuted == v
    end
    q .+= -0.5 .* v
    @test q == v / 2
    @inbounds q[1] = q[1] + 1
    @test q[1] == 1.5
    @test q[CartesianIndex(2)] == 1
    @test sum(q) == 6
    @test dot(q, q) == sum(abs2, q)
    @test q .+ 1 == collect(q) .+ 1
    @test similar(q) isa Vector{Float64}
    @test similar(q, Int, (2, 3)) isa Matrix{Int}
    saved = copy(q)
    fill!(q, 9)
    @test all(==(9), q)
    @test saved == [1.5, 1, 1.5, 2]
    @test_throws BoundsError q[0]
    @test_throws BoundsError q[5] = 1
    # Standard Julia copying/broadcasting rejects incompatible sizes before writing.
    @test_throws DimensionMismatch q .= zeros(3)
    @test_throws BoundsError copyto!(q, zeros(5))
    @test all(==(9), q)
    short = fill(-1., 3)
    @test_throws BoundsError copyto!(short, q)
    @test_throws DimensionMismatch short .= q
    @test short == fill(-1., 3)
    q .= [2.0]  # singleton broadcast expansion
    @test q == fill(2., 4)
    larger = fill(-1., 5)
    copyto!(larger, q)
    @test larger == [2, 2, 2, 2, -1]
    copyto!(q, [7., 8.])  # copy only the source length
    @test q == [7, 8, 2, 2]
    # Fused functions, scalar expansion, and direct lazy-broadcast copies.
    q .= 2 .* sin.(v) .+ 1
    @test q ≈ 2 .* sin.(v) .+ 1
    broadcast!(+, q, v, 2)
    @test q == v .+ 2
    copyto!(q, Base.Broadcast.broadcasted(*, v, 3))
    @test q == 3v

    # Ordinary self-broadcast needs no alias-protection buffer.
    copyto!(q, v)
    q .*= 2
    @test q == 2v
    copyto!(q, v)

    # Aliasing is detected, but callers must explicitly copy overlapping sources.
    source = view(siteview(state, I, tile), 4:-1:1)
    @test Base.mightalias(q, source)
    @test_throws ArgumentError q .= source
    @test q == v
    q .= copy(source)
    @test q == reverse(v)
    copyto!(q, v)
    @test_throws ArgumentError copyto!(view(q, 2:4), view(q, 1:3))
    @test q == v
    copyto!(view(q, 2:4), copy(view(q, 1:3)))
    @test q == [1, 1, 2, 3]
    q .= reverse(q)
    @test q == [3, 2, 1, 1]
    site_allocations(state, I, tile, out)
    @test site_allocations(state, I, tile, out) == (0, 0, 0, 0)
end

@testset "Site vectors" begin
    flat_state = ArrayOfHaloArray(LocalHaloArray, Float64, (4,), (3,2), 1;
                                  boundary_condition=:periodic)
    exercise_siteview(flat_state)
    local_state = ArrayOfHaloArray(LocalHaloArray, Float64, (2,2), (3,2), 1;
                                   boundary_condition=:periodic)
    I = first(interior_cells(CellRanges(local_state)))
    q = siteview(local_state, I)
    q .= [11. 21.; 12. 22.]
    @test parent(local_state[1,1])[I] == 11
    @test parent(local_state[2,1])[I] == 12
    @test parent(local_state[1,2])[I] == 21
    @test parent(local_state[2,2])[I] == 22
    @test siteview(local_state, I, 1) == q
    synchronize_halo!(local_state)
    @test siteview(local_state, CartesianIndex(5,2)) == q
    parent(local_state[1,1])[I] = 42
    @test q[1] == 42

    named = MultiHaloArray(LocalHaloArray, Float64, (3,2), 1;
        fields=(:a,:b,:c,:d), boundary_condition=:periodic)
    exercise_siteview(named)
    @test siteview(named, I)[2] == parent(named.b)[I]

    # Broadly typed flat containers remain valid.
    broad_fields = Any[named.a, named.b]
    broad = ArrayOfHaloArray(broad_fields)
    @test collect(siteview(broad, I)) == [parent(named.a)[I], parent(named.b)[I]]

    threaded = ArrayOfHaloArray(ThreadedHaloArray, Float64, (4,), (3,2), 1;
        dims=(2,1), boundary_condition=:periodic)
    exercise_siteview(threaded, 2)
    @test all(iszero, siteview(threaded, I, 1))

    topology = CartesianTopology(MPI.COMM_SELF, (1,1); periodic=(true,true))
    distributed = ArrayOfHaloArray(HaloArray, Float64, (3,2), 1, topology;
        boundary_conditions=fill(:periodic,4))
    exercise_siteview(distributed)

    # Shared fields in a different collection/order must be recognized as aliases.
    copyto!(q, [1, 2, 3, 4])
    reordered = ArrayOfHaloArray(reshape(reverse(vec(parent(local_state))), 2, 2))
    @test_throws ArgumentError q .= siteview(reordered, I)
    @test vec(q) == [1, 2, 3, 4]
    q .= copy(siteview(reordered, I))
    @test vec(q) == [4, 3, 2, 1]

    a = LocalHaloArray(Int, (3,2), 1; boundary_condition=:periodic)
    b = LocalHaloArray(Float64, (3,2), 1; boundary_condition=:periodic)
    mixed = siteview(MultiHaloArray((; a, b)), I)
    mixed .= [2., 3.5]
    @test mixed[1] === 2.0
    @test mixed[2] === 3.5
    @test eltype(mixed) === Float64
    @test all(x -> x isa eltype(mixed), mixed)
    @test map(identity, mixed) isa Vector{Float64}
    @test [x for x in mixed] isa Vector{Float64}
    @test Base.return_types(getindex, Tuple{typeof(mixed),Int}) == [Float64]
    @test copy(mixed) isa Vector{Float64}
    @test copy(mixed) == [2., 3.5]
    @test parent(a)[I] === 2
    @test_throws InexactError mixed[1] = 1.5

    # The component count remains runtime-sized, including large collections.
    types = []
    for n in (8, 64, 256, 1024)
        state = ArrayOfHaloArray(LocalHaloArray, Float64, (n,), (3,2), 1;
            boundary_condition=:periodic)
        qn = siteview(state, I)
        push!(types, typeof(qn))
        qn .= 1:n
        @test sum(qn) == n * (n + 1) / 2
        buffer = zeros(n)
        site_allocations(state, I, nothing, buffer)
        @test site_allocations(state, I, nothing, buffer) == (0, 0, 0, 0)
    end
    @test all(==(first(types)), types)
    @test !isdefined(HaloArrays, :gather_fields!)
    @test !isdefined(HaloArrays, :scatter_fields!)
    @test !isdefined(HaloArrays, :add_fields!)
end

@testset "Site matrices" begin
    U = ArrayOfHaloArray(LocalHaloArray, Float64, (2,2), (3,2,2), 1;
        boundary_condition=:periodic)
    I = first(interior_cells(CellRanges(U)))
    m = @inferred siteview(U, I)
    @test m isa AbstractMatrix{Float64}
    @test size(m) == field_shape(U) == (2, 2)
    @test IndexStyle(typeof(m)) == IndexLinear()
    M = [1. 3.; 2. 4.]
    m .= M
    @test all(parent(U[a, b])[I] == M[a, b] for a in 1:2, b in 1:2)
    @test m == M
    @test m[2, 1] == 2 && m[3] == 3
    m[1, 2] = 30
    @test parent(U[1, 2])[I] == 30
    @test_throws BoundsError m[3, 1]

    # Flat buffers: copyto! is linear (column-major); broadcasts need the site shape.
    flat = zeros(4)
    @test copyto!(flat, m) == [1, 2, 30, 4]
    copyto!(m, [5., 6., 7., 8.])
    @test m == [5. 7.; 6. 8.]
    @test_throws DimensionMismatch m .= [1., 2., 3., 4.]
    vec(m) .= [1., 2., 3., 4.]
    @test m == M
    m .+= 0.5 .* M
    @test m == 1.5 .* M
    @test m * [1., 1.] == (1.5 .* M) * [1., 1.]
    @test copy(m) isa Matrix{Float64}
    @test copy(m) == m
    @test similar(m) isa Matrix{Float64}
    @test size(similar(m)) == (2, 2)

    # A transposed view of the same site overlaps and needs an explicit copy.
    @test_throws ArgumentError m .= transpose(m)
    @test m == 1.5 .* M
    m .= copy(transpose(m))
    @test m == 1.5 .* permutedims(M)

    site_allocations(U, I, nothing, zeros(2, 2))
    @test site_allocations(U, I, nothing, zeros(2, 2)) == (0, 0, 0, 0)

    tiled = ArrayOfHaloArray(ThreadedHaloArray, Float64, (2,2), (3,2), 1;
        dims=(2,1), boundary_condition=:periodic)
    J = CartesianIndex(2, 2)
    mt = siteview(tiled, J, 2)
    @test size(mt) == (2, 2)
    mt .= M
    @test all(tile_parent(tiled[a, b], 2)[J] == M[a, b] for a in 1:2, b in 1:2)
    @test all(iszero, siteview(tiled, J, 1))
end

@testset "Single-field site vectors" begin
    single = LocalHaloArray(Float64, (3,2), 1; boundary_condition=:periodic)
    threaded = ThreadedHaloArray(Float64, (3,2), 1;
        dims=(2,1), boundary_condition=:periodic)
    topology = CartesianTopology(MPI.COMM_SELF, (1,1); periodic=(true,true))
    distributed = HaloArray(Float64, (3,2), 1, topology; boundary_condition=:periodic)
    I = first(interior_cells(CellRanges(single)))

    for (state, tile) in ((single, nothing), (threaded, 2), (distributed, nothing))
        q = @inferred siteview(state, I, tile)
        @test q isa AbstractVector{Float64}
        @test parent(q) === state
        @test size(q) == (1,)
        @test axes(q) == (Base.OneTo(1),)
        q[1] = 3
        @test (@inferred q[1]) === 3.0
        @test tile_parent(state, something(tile, 1))[I] == 3
        tile_parent(state, something(tile, 1))[I] = 4
        @test q[1] == 4
        q .*= 2
        @test q[1] == 8
        out = zeros(1)
        @test copyto!(out, q) === out
        @test out == [8]
        copyto!(q, [5.])
        @test sum(q) == 5
        saved = copy(q)
        q[1] = 7
        @test saved == [5]
        @test similar(q) isa Vector{Float64}
        @test_throws BoundsError q[0]
        @test_throws BoundsError q[2] = 1
        @test_throws DimensionMismatch q .= [1., 2.]
        @test_throws BoundsError copyto!(q, [1., 2.])
        @test q[1] == 7
        @test Base.mightalias(q, tile_parent(state, something(tile, 1)))
        @test Base.dataids(q) == Base.dataids(tile_parent(state, something(tile, 1)))
        # Alias detection works between a single field and a collection containing it.
        collection_view = siteview(MultiHaloArray((; a=state)), I, tile)
        @test Base.mightalias(q, collection_view)
        @test Base.mightalias(collection_view, q)
        @test_throws ArgumentError q .= collection_view
        q .= copy(collection_view)
        @test q[1] == 7
        site_allocations(state, I, tile, out)
        @test site_allocations(state, I, tile, out) == (0, 0, 0, 0)
    end

    @test siteview(single, I, 1) == siteview(single, I)
    @test siteview(threaded, I, 1)[1] == 0
    @test_throws ArgumentError siteview(threaded, I)[1]
    @test_throws BoundsError siteview(threaded, I, 3)[1]
    @test_throws BoundsError siteview(single, I, 2)[1]
    @test_throws BoundsError siteview(single, CartesianIndex(0,2))[1]
    synchronize_halo!(single)
    @test siteview(single, CartesianIndex(5,2))[1] == siteview(single, I)[1]
    siteview(single, CartesianIndex(1,2))[1] = -3
    @test parent(single)[1,2] == -3

    # A structured cell is still a single field, with no implicit flattening.
    vector_cell = LocalHaloArray(SVector{2,Float64}, (3,2), 1; boundary_condition=:periodic)
    q = siteview(vector_cell, I)
    q[1] = SVector(2., 3.)
    @test length(q) == 1
    @test eltype(q) === SVector{2,Float64}
    @test q[1] === SVector(2., 3.)
    @test parent(vector_cell)[I] === SVector(2., 3.)
    @test copy(q) == [SVector(2., 3.)]
end
