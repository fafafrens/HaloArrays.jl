using MPI
using BenchmarkTools
using HaloArrays

# using your HaloArray module here, assumed imported or defined in same file

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Setup
dims = (0,0,0)  # adjust as needed
bd=((Periodic(),Periodic()),(Periodic(),Periodic()),(Periodic(),Periodic()))  # periodic boundary conditions
topo = CartesianTopology(comm, dims)
halo = HaloArray(Float64, (22,22,10), 1, topo, bd)

function time_exchange(name, f, halo; reps=10000)
    times = zeros(reps)
    for i in 1:reps
        MPI.Barrier(MPI.COMM_WORLD)
        t = @elapsed f(halo)
        MPI.Barrier(MPI.COMM_WORLD)
        times[i] = t
    end
    avg_time = mean(times)
    if rank == 0
        println("[$name] Average time over $reps runs: $(round(avg_time*1e3, digits=5)) ms")
    end
end

# Warm up
#halo_exchange!(halo)
#halo_exchange_wait!(halo)
#println("Rank $rank halo_exchange_wait!")
halo_exchange_waitall!(halo)
println("Rank $rank halo_exchange_waitall!")
halo_exchange_async!(halo)
println("Rank $rank halo_exchange_async!")
halo_exchange_waitall_unsafe!(halo)
println("Rank $rank halo_exchange_waitall_unsafe!")
halo_exchange_async_unsafe!(halo)
println("Rank $rank halo_exchange_async_unsafe!")



#@benchmark halo_exchange!($halo)
#@benchmark halo_exchange_wait!($halo)
#@benchmark halo_exchange_waitall!($halo)
#@benchmark halo_exchange_async!($halo)
#@benchmark halo_exchange_waitall_unsafe!($halo)
#@benchmark halo_exchange_async_unsafe!($halo)


#@code_warntype halo_exchange_waitall_unsafe!(halo)

#using Profile

#Profile.Allocs.clear()

#@time Profile.Allocs.@profile sample_rate=1 halo_exchange_async!(halo)

#using PProf

#PProf.Allocs.pprof(from_c=false)

#@benchmark get_recv_view(Side(2), Dim(2), $halo)


# Run benchmark
if rank == 0
    println("== Benchmarking Halo Exchange Variants ==")
end

#time_exchange("Naive Test Loop", halo_exchange!, halo)
#time_exchange("Wait + Wait", halo_exchange_wait!, halo)
time_exchange("Waitall", halo_exchange_waitall!, halo)
time_exchange("Async", halo_exchange_async!, halo)
time_exchange("Waitall Unsafe", halo_exchange_waitall_unsafe!, halo)
time_exchange("Async Unsafe", halo_exchange_async_unsafe!, halo)

MPI.Barrier(comm)
