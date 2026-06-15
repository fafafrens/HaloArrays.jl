# Arrays, layout & reductions

Construction, ownership/layout accessors, indexing, reductions, thread backends,
`gather`, and HDF5 output.

```@autodocs
Modules = [HaloArrays]
Pages = ["haloarray.jl", "abstract_haloarray.jl", "local_haloarray.jl",
         "threaded_haloarray.jl", "thread_backend.jl", "reduction.jl",
         "vector_space.jl", "gather.jl", "save_hdf5.jl", "mpi_support.jl",
         "cartesian_topology.jl", "field_collection.jl", "multihaloarray.jl",
         "ArrayOfHaloArray.jl", "maybehaloarray.jl"]
Order = [:function]
Private = false
```
