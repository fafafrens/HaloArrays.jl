using Test
using MPI
using HaloArrays

if !MPI.Initialized()
    MPI.Init()
end

function _test_topology(dims::NTuple{N,Int}) where {N}
    return CartesianTopology(MPI.COMM_SELF, dims; periodic=ntuple(_ -> false, Val(N)))
end

@testset "broadcast styles follow Base's constructor convention" begin
    # Style{M}(Val(N)) must build Style{N}: Base's result_style re-dimensions a
    # style that way when it wins over another one. Before, only N == M was
    # defined, so a collection (D = spatial + 1) mixed with a single array died
    # in a MethodError.
    for S in (HaloArrays.HaloArrayStyle, HaloArrays.ThreadedHaloArrayStyle,
              HaloArrays.MultiHaloArrayStyle, HaloArrays.MaybeHaloArrayStyle)
        @test S{2}(Val(3)) === S{3}()
        @test S(Val(2)) === S{2}()
    end
    u = LocalHaloArray(Float64, (4, 3), 1; boundary_condition=:periodic)
    fill!(u, 2.0)
    m = MultiHaloArray(LocalHaloArray, Float64, (4, 3), 1; fields=(:a, :b), boundary_condition=:periodic)
    fill!(m, 1.0)
    w = similar(m)
    w .= m .+ u                       # the single array applies to every field
    @test all(==(3.0), interior_view(w.a)) && all(==(3.0), interior_view(w.b))
    w .= u .* m .- 1
    @test all(==(1.0), interior_view(w.b))
    # Out of place, Base's instantiate rejects the 3-D/2-D operand axes before
    # any HaloArrays code runs: only the in-place form is supported.
    @test_throws DimensionMismatch m .+ u
    # The same on the threaded backend (its style needs its own rule against collections).
    t = ThreadedHaloArray(Float64, (2, 3), 1; dims=(2, 1), boundary_condition=:periodic)
    fill!(t, 2.0)
    tm = MultiHaloArray(ThreadedHaloArray, Float64, (2, 3), 1; dims=(2, 1), fields=(:a, :b), boundary_condition=:periodic)
    fill!(tm, 1.0)
    tw = similar(tm)
    tw .= tm .+ t
    @test all(==(3.0), gather_haloarray(tw.a)) && all(==(3.0), gather_haloarray(tw.b))
    tw .= t .* tm .- 1
    @test all(==(1.0), gather_haloarray(tw.b))
    @test_throws DimensionMismatch tm .+ t
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

    fields = MultiHaloArray((; u, v))

    @test fields isa MultiHaloArray
    @test fields isa AbstractArray{Float64,3}
    @test eltype(fields) === Float64
    @test ndims(fields) == 3
    @test HaloArrays.n_field(fields) == 2
    @test fields[:u] === u
    @test fields[:v] === v
    @test eltype(typeof(fields)) === Float64
    @test is_active(fields)

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

    @test length(fields) == prod(size(fields))
    @test first(eachindex(fields)) == CartesianIndex(1, 1, 1)

    copied_into = similar(fields)
    fill!(copied_into, -1)
    @test copyto!(copied_into, fields) === copied_into
    @test collect(interior_view(copied_into.arrays.u)) == collect(interior_view(fields.arrays.u))
    @test collect(interior_view(copied_into.arrays.v)) == collect(interior_view(fields.arrays.v))

    zero_fields = zero(fields)
    @test zero_fields isa MultiHaloArray
    @test all(==(0), zero_fields)
    @test fill!(zero_fields, 7) === zero_fields
    @test all(==(7), zero_fields)

    from_bcs = MultiHaloArray(HaloArray, Float64, (3, 2), 1, topology;
        boundary_conditions=(; rho=:repeating, mom=:repeating))
    @test from_bcs isa MultiHaloArray
    @test from_bcs[:rho] isa HaloArray
    @test size(from_bcs) == (2, 3, 2)
    @test size(from_bcs) == size(from_bcs)
    @test axes(from_bcs) == map(Base.OneTo, size(from_bcs))
    @test interior_axes(from_bcs) == map(Base.OneTo, interior_size(from_bcs))
    @test interior_size(from_bcs) == (2, 3, 2)
    @test size(from_bcs) == (2, 3, 2)
    @test eltype(from_bcs) === Float64

    local_fields = MultiHaloArray(LocalHaloArray, Int, (3,), 1;
        boundary_conditions=(; rho=:repeating, mom=:antireflecting))
    interior_view(local_fields.arrays.rho) .= [1, 2, 3]
    interior_view(local_fields.arrays.mom) .= [10, 20, 30]

    @test local_fields isa MultiHaloArray
    @test local_fields isa AbstractArray{Int,2}
    @test local_fields[:rho] isa LocalHaloArray
    @test size(local_fields) == (2, 3)
    @test size(local_fields) == size(local_fields)
    @test interior_axes(local_fields) == map(Base.OneTo, interior_size(local_fields))
    @test interior_size(local_fields) == (2, 3)
    @test local_fields[1] === local_fields.arrays.rho
    @test local_fields[1, 2] == 2
    local_fields[2, 3] = 35
    @test local_fields[2, 3] == 35
    @test interior_view(local_fields.arrays.mom)[3] == 35
    local_fields[2, 3] = 30

    shifted_local = local_fields .+ 4
    @test shifted_local isa MultiHaloArray
    @test shifted_local.arrays.rho isa LocalHaloArray
    @test collect(interior_view(shifted_local.arrays.rho)) == [5, 6, 7]
    @test collect(interior_view(shifted_local.arrays.mom)) == [14, 24, 34]

    local_dest = similar(local_fields)
    local_dest .= 2 .* local_fields .+ shifted_local
    @test collect(interior_view(local_dest.arrays.rho)) == [7, 10, 13]
    @test collect(interior_view(local_dest.arrays.mom)) == [34, 64, 94]

    resized_local = similar(local_fields, Float32, (2, 5))
    @test resized_local isa MultiHaloArray
    @test eltype(resized_local) === Float32
    @test size(resized_local) == (2, 5)
    @test interior_size(resized_local) == (2, 5)
    @test_throws DimensionMismatch similar(local_fields, Float32, (3, 5))

    synchronize_halo!(local_fields)
    @test parent(local_fields.arrays.rho) == [1, 1, 2, 3, 3]
    @test parent(local_fields.arrays.mom) == [-10, 10, 20, 30, -30]

    # parent(collection) is the one-level unwrap: the NamedTuple of *fields*.
    # field_storages(collection) pushes parent to the leaves: the NamedTuple of
    # raw padded storages (what FV stencils index with ghost offsets).
    @test parent(local_fields) === local_fields.arrays
    @test parent(local_fields).rho isa LocalHaloArray
    @test field_storages(local_fields).rho === parent(local_fields.arrays.rho)
    @test keys(field_storages(local_fields)) == keys(local_fields.arrays)

    # field_storages! also fills a named container, in declaration order.
    named_cache = collect(field_storages(local_fields))
    @test field_storages!(named_cache, local_fields) === named_cache
    @test Tuple(named_cache) == Tuple(field_storages(local_fields))

    threaded_fields = MultiHaloArray(ThreadedHaloArray, Int, (3,), 1;
        dims=(2,),
        boundary_conditions=(; rho=:repeating, mom=:repeating))
    interior_view(threaded_fields.arrays.rho, 1) .= [1, 2, 3]
    interior_view(threaded_fields.arrays.rho, 2) .= [4, 5, 6]
    interior_view(threaded_fields.arrays.mom, 1) .= [10, 20, 30]
    interior_view(threaded_fields.arrays.mom, 2) .= [40, 50, 60]

    @test threaded_fields isa MultiHaloArray
    @test threaded_fields isa AbstractArray{Int,2}
    @test threaded_fields[:rho] isa ThreadedHaloArray
    @test size(threaded_fields) == (2, 6)
    @test size(threaded_fields) == size(threaded_fields)
    @test interior_axes(threaded_fields) == map(Base.OneTo, interior_size(threaded_fields))
    @test interior_size(threaded_fields) == (2, 6)

    shifted_threaded = threaded_fields .+ 3
    @test shifted_threaded isa MultiHaloArray
    @test shifted_threaded.arrays.rho isa ThreadedHaloArray
    @test collect(interior_view(shifted_threaded.arrays.rho, 1)) == [4, 5, 6]
    @test collect(interior_view(shifted_threaded.arrays.rho, 2)) == [7, 8, 9]
    @test collect(interior_view(shifted_threaded.arrays.mom, 1)) == [13, 23, 33]
    @test collect(interior_view(shifted_threaded.arrays.mom, 2)) == [43, 53, 63]

    threaded_dest = similar(threaded_fields)
    threaded_dest .= threaded_fields .+ shifted_threaded
    @test collect(interior_view(threaded_dest.arrays.rho, 1)) == [5, 7, 9]
    @test collect(interior_view(threaded_dest.arrays.rho, 2)) == [11, 13, 15]
    @test collect(interior_view(threaded_dest.arrays.mom, 1)) == [23, 43, 63]
    @test collect(interior_view(threaded_dest.arrays.mom, 2)) == [83, 103, 123]

    threaded_copy = similar(threaded_fields)
    @test copyto!(threaded_copy, threaded_fields) === threaded_copy
    for name in keys(threaded_fields.arrays), tile_id in 1:tile_count(threaded_fields)
        @test tile_parent(threaded_copy.arrays[name], tile_id) ==
              tile_parent(threaded_fields.arrays[name], tile_id)
    end

    threaded_zero = zero(threaded_fields)
    @test threaded_zero isa MultiHaloArray
    @test fill!(threaded_zero, -3) === threaded_zero
    # fill! is interior-only; ghosts are refreshed by synchronize_halo!
    for field in values(threaded_zero.arrays), tile_id in 1:tile_count(threaded_zero)
        @test all(==(-3), interior_view(field, tile_id))
    end

    resized_threaded = similar(threaded_fields, Float32, (2, 8))
    @test resized_threaded isa MultiHaloArray
    @test eltype(resized_threaded) === Float32
    @test size(resized_threaded) == (2, 8)
    @test tile_size(resized_threaded) == (4,)
    @test_throws DimensionMismatch similar(threaded_fields, Float32, (3, 8))

    synchronize_halo!(threaded_fields)
    @test tile_parent(threaded_fields.arrays.rho, 1) == [1, 1, 2, 3, 4]
    @test tile_parent(threaded_fields.arrays.rho, 2) == [3, 4, 5, 6, 6]
    @test tile_parent(threaded_fields.arrays.mom, 1) == [10, 10, 20, 30, 40]
    @test tile_parent(threaded_fields.arrays.mom, 2) == [30, 40, 50, 60, 60]

    q_arrays = [HaloArray(Float64, (3, 2), 1, topology; boundary_condition=:repeating) for _ in 1:2]
    for c in eachindex(q_arrays)
        q_interior = interior_view(q_arrays[c])
        for i in 1:size(q_interior, 1), j in 1:size(q_interior, 2)
            q_interior[i, j] = 100 * c + 10 * i + j
        end
    end
    q = ArrayOfHaloArray(q_arrays)

    # A leaf next to a collection, or collections of different field shapes,
    # cannot form one rectangular array: rejected.
    @test_throws DimensionMismatch MultiHaloArray((; rho=u, q))
    @test_throws DimensionMismatch MultiHaloArray((; a=q, b=ArrayOfHaloArray([copy(u), copy(u), copy(u)])))

    # Nested collections of equal field shape: one array whose field axes are
    # the outer container's followed by the inner ones.
    p = ArrayOfHaloArray([copy(u), 2 .* u])
    nested_fields = MultiHaloArray((; q, p))
    @test nested_fields isa MultiHaloArray
    @test eltype(nested_fields) == Float64
    @test ndims(nested_fields) == 4
    @test size(nested_fields) == (2, 2, 3, 2)
    @test size(nested_fields, 4) == 2 && length(nested_fields) == 24
    @test field_shape(nested_fields) == (2, 2) && HaloArrays.n_field(nested_fields) == 4
    @test interior_size(nested_fields) == (2, 2, 3, 2)
    @test storage_size(nested_fields) == (2, 2, 5, 4)
    @test axes(nested_fields) == (Base.OneTo(2), Base.OneTo(2), Base.OneTo(3), Base.OneTo(2))
    @test halo_width(nested_fields) == 1
    @test nested_fields[:q] === q && nested_fields.p === p
    # full indexing: outer, inner, spatial — and the same through siteview
    @test nested_fields[1, 2, 3, 1] == q[2][3, 1] == 231
    @test nested_fields[2, 1, 1, 2] == u[1, 2]
    nested_fields[2, 2, 1, 1] = 7.5
    @test p[2][1, 1] == 7.5
    sv = siteview(nested_fields, CartesianIndex(2, 2))   # storage index of cell (1, 1)
    @test size(sv) == (2, 2)
    @test sv == [nested_fields[i, j, 1, 1] for i in 1:2, j in 1:2]
    sv[1, 1] = 42.0
    @test q[1][1, 1] == 42.0
    # fill_from_global_indices!: one value per leaf, column-major field order
    fill_from_global_indices!(I -> (1.0, 2.0, 3.0, 4.0), nested_fields)
    @test all(==(1.0), interior_view(q[1])) && all(==(2.0), interior_view(p[1]))
    @test all(==(3.0), interior_view(q[2])) && all(==(4), interior_view(p[2]))
    fill_from_global_indices!(I -> I[1] + 10 * I[2], nested_fields)
    @test p[2][3, 2] == 23 && q[2][3, 2] == 23
    @test sum(nested_fields) == 4 * sum(i + 10j for i in 1:3, j in 1:2)

    nested_shifted = nested_fields .+ 2
    @test nested_shifted isa MultiHaloArray
    @test nested_shifted.arrays.q isa ArrayOfHaloArray
    @test size(nested_shifted) == size(nested_fields)
    @test collect(interior_view(nested_shifted.arrays.q[1])) == [i + 10j + 2 for i in 1:3, j in 1:2]

    nested_dest = similar(nested_fields)
    nested_dest .= 2 .* nested_fields .+ nested_shifted
    @test collect(interior_view(nested_dest.arrays.p[2])) == [3 * (i + 10j) + 2 for i in 1:3, j in 1:2]

    # dims reductions: a spatial axis keeps both field axes; a field axis of a
    # nested collection is refused
    r = HaloArrays.getdata(sum(nested_fields; dims=3))     # MPI fields: Maybe-wrapped like a flat collection
    @test r isa MultiHaloArray && size(r) == (2, 2, 2)
    @test r[1, 2, 1] == sum(q[2][i, 1] for i in 1:3)
    @test_throws ArgumentError sum(nested_fields; dims=1)
    @test_throws ArgumentError sum(nested_fields; dims=2)

    synchronize_halo!(nested_fields)
    @test parent(nested_fields.arrays.q[1])[1, 2] == first(interior_view(nested_fields.arrays.q[1]))
    @test all(x -> x > 0, nested_fields)
    @test any(x -> x == 23, nested_fields)
    g = gather_haloarray(nested_fields)
    @test keys(g) == (:q, :p) && size(g.q) == (2, 3, 2)
    # an ArrayOfHaloArray of ArrayOfHaloArrays gathers to one array, outer axis first
    aa = ArrayOfHaloArray([q, p])
    @test ndims(aa) == 4 && HaloArrays._spatial_ndims(aa) == 2 && field_shape(aa) == (2, 2)
    ga = gather_haloarray(aa)
    @test size(ga) == (2, 2, 3, 2)
    @test ga[1, 2, 3, 1] == q[2][3, 1] && ga[2, 1, 1, 2] == p[1][1, 2]
    @test size(siteview(aa, CartesianIndex(2, 2))) == (2, 2)
    @test size(HaloArrays.getdata(sum(aa; dims=3))) == (2, 2, 2)      # spatial axis, not the inner field axis
    # similar with explicit dims: container axes first, then the field's own dims
    @test size(similar(nested_fields, Float32, (2, 2, 3, 2))) == (2, 2, 3, 2)
    @test size(similar(nested_fields, (2, 2, 6, 4))) == (2, 2, 6, 4)
    @test_throws DimensionMismatch similar(nested_fields, (3, 2, 3, 2))   # named outer cannot grow
    @test size(similar(aa, Float64, (3, 2, 3, 2))) == (3, 2, 3, 2)      # array outer can
    @test size(similar(aa, (2, 3, 3, 2))) == (2, 3, 3, 2)               # and so can the inner array containers
    # raw storages: nested containers down to the leaves; the flat refill takes
    # one leaf per entry in column-major field order
    fs = field_storages(nested_fields)
    @test keys(fs) == (:q, :p) && fs.q[2] === parent(q[2]) && fs.p[1] === parent(p[1])
    cache = Vector{Matrix{Float64}}(undef, 4)
    @test field_storages!(cache, nested_fields) === cache
    @test cache[1] === parent(q[1]) && cache[2] === parent(p[1]) && cache[4] === parent(p[2])
    @test_throws DimensionMismatch field_storages!(Vector{Matrix{Float64}}(undef, 2), nested_fields)
    # alias checks descend to the leaves: self-aliased site operations are detected
    sq = siteview(nested_fields, CartesianIndex(2, 2))
    @test Base.mightalias(sq, parent(q[1])) && !Base.mightalias(sq, zeros(2))
    @test Base.mightalias(sq, siteview(nested_fields, CartesianIndex(3, 2)))
    sq .= sq .+ 1                               # broadcast with alias preprocessing
    @test sq[1, 1] == q[1][1, 1]
    geo = cell_geometry(nested_fields, UniformAxis(0, 1), UniformAxis(0, 1))
    @test size(geo) == (4, 3, 2)

    copied = copy(fields)
    interior_view(copied.arrays.u)[1, 1] = -1
    @test interior_view(fields.arrays.u)[1, 1] != interior_view(copied.arrays.u)[1, 1]

    # fields + boundary_condition shorthand
    from_fields = MultiHaloArray(HaloArray, Float64, (3, 2), 1, topology;
        fields=(:a, :b, :c), boundary_condition=:repeating)
    @test from_fields isa MultiHaloArray
    @test keys(from_fields.arrays) == (:a, :b, :c)
    @test all(f -> f isa HaloArray, values(from_fields.arrays))
    @test size(from_fields) == (3, 3, 2)

    from_fields_default_type = MultiHaloArray(HaloArray, (3, 2), 1, topology;
        fields=(:x, :y), boundary_condition=:repeating)
    @test eltype(from_fields_default_type) === Float64

    bad = HaloArray(Float64, (4, 2), 1, topology; boundary_condition=:repeating)
    @test_throws DimensionMismatch MultiHaloArray((; u, bad))

    bad_q = ArrayOfHaloArray([HaloArray(Float64, (4, 2), 1, topology; boundary_condition=:repeating)])
    @test_throws DimensionMismatch MultiHaloArray((; u, q=bad_q))

    @test all(x -> x > 0, fields)
    @test any(x -> x == 22, fields)
end
