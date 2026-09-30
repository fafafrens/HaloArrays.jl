# HDF5 I/O — public API stub. The methods live in the `HaloArraysHDF5Ext` package
# extension, which loads only when the user has `using HDF5`. This keeps HDF5 (and
# its MPI-built JLLs, which clash with a system CUDA-aware MPI) out of the core
# dependency tree. Without HDF5 loaded the call throws a MethodError, by design.

"""
    append_haloarray!(file_or_group, name, u) -> dataset | group | nothing
    append_haloarray!(filename, name, u)

Append the interior (ghost-free) data of `u` as the next step of the time-series
dataset `name`: the dataset has time on its leading axis, is created on the
first call and grows by one slab per call, and an existing dataset is validated
(shape and element type) before it is reused. A single halo array or an
[`ArrayOfHaloArray`](@ref) is one dataset (field axes first for the latter); a
[`MultiHaloArray`](@ref) is a group with one such dataset per field. A snapshot
is a single append.

With an `HDF5.File` or `HDF5.Group` the caller owns the file: open it once for
the whole run, and open it collectively (`h5open(path, mode, communicator(u),
MPI.Info())`) for a distributed `HaloArray`, since every rank writes its own
block. The dataset or group is returned, e.g. to attach attributes. With a
`filename` the file is opened and closed on each call, on the array's own
communicator, which is the convenient form for a [`MaybeHaloArray`](@ref)
(a `dims=` reduction result living on a sub-communicator); an inactive
`MaybeHaloArray` is a no-op. The name is used as given.

To write a gathered global array instead (no parallel HDF5 needed), use
[`gather_haloarray`](@ref) with plain HDF5.jl:
```julia
A = gather_haloarray(u)
is_root(u) && h5write("snapshot.h5", "rho", A)
```

Requires `using HDF5` (provided by the `HaloArraysHDF5Ext` extension).

```julia
h5open("run.h5", "w", communicator(u), MPI.Info()) do file
    for step in 1:nsteps
        advance!(u)
        dset = append_haloarray!(file, "rho", u)
        attributes(dset)["t"] = step * dt
    end
end
profile = sum(u; dims=3)
append_haloarray!("profile.h5", "profile", profile)
```
"""
function append_haloarray! end
