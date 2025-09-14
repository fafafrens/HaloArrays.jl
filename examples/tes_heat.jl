using MPI
using HDF5
using HaloArrays


function heat_1d_step!( u_old::HaloArray, α, dt, dx)
    
    data_old=u_old.data

@inbounds for I in CartesianIndices(HaloArrays.interior_range(u_old))
        # Because of halos, interior indices shifted by h
        idx=first(Tuple(I))

        # Discrete Laplacian using halo data
        laplacian = (data_old[idx-1] - 2*data_old[idx] + data_old[idx+1]) / dx^2
        

        data_old[I] =  data_old[I] + α * dt * laplacian

    end
    return nothing
end


function run_heat_1d_periodic(N_local, α, CFL, L, nt)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # 1D periodic domain setup
    dims = (nproc,)
    periods = (true,)  # periodic boundary in 1D


    Nglobal = N_local * nproc

    dx = L / Nglobal  # spatial step size
    dt = 0.5 * dx^2 / α
    h = 1
    u= HaloArray(Float64,(N_local,),1,((Periodic(),Periodic()),))
   

    # Initialize Gaussian pulse across global domain

HaloArrays.fill_from_global_indices!(u) do i
    x = i[1]
    return   exp(-(x - Nglobal/2)^2/10^2)+1
    end
    
    gather_and_append_haloarray!("result","temp",u; root=0)
    append_haloarray_to_file!("result_par","temp",u)
    #fid ,dset=create_haloarray_output_file("result_par_fix.h5", "temp", u, nt+1)
    #write_haloarray_timestep!(dset, u, 1)  # Write initial condition

    MPI.Barrier(comm)
    for step in 1:nt
        # Halo exchange updates ghost cells, periodic neighbors wrap correctly
        halo_exchange_async_unsafe!(u)

        if any(isnan, u.data)
            @show step, u.data
            error("Trovato NaN dopo halo_exchange!")
        end

        # Compute heat equation step
        heat_1d_step!(u, α, dt, dx)
        
       # --- Timing the serial write ---
    t_serial_start = MPI.Wtime()
    gather_and_append_haloarray!("result", "temp", u; root=0)
    t_serial_end = MPI.Wtime()
    t_serial = t_serial_end - t_serial_start

    # --- Timing the parallel write ---
    t_par_start = MPI.Wtime()
    append_haloarray_to_file!("result_par", "temp", u)
    t_par_end = MPI.Wtime()
    t_par = t_par_end - t_par_start

    # Only rank 0 prints the timings
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @info "Step $step write times — serial: $(round(t_serial*1e3, digits=3)) ms, parallel: $(round(t_par*1e3, digits=3)) ms"
    end

         MPI.Barrier(comm)
    end

    #close(fid)  # Close the HDF5 file
    
    if rank == 0
        println("Heat equation simulation completed.")
        
    end
    

    #return gather_haloarray(u)
        MPI.Finalize()
end


N_local=100
L = 1.0
CFL= 0.5  # CFL condition for stability

alpha = 1.0  # thermal diffusivity
nt = 100 # number of time steps

run_heat_1d_periodic(N_local, alpha, CFL, L, nt)


file = h5open("result.h5") 
file2 = h5open("result_par.h5") 

dset = file["temp"]
dset2 = file2["temp"]

size(dset)
size(dset2)

using Plots
plot(dset[1,:], label="t=0")
plot!(dset2[1,:], label="t=0")
plot!(dset[2,:], label="t=1")
plot!(dset2[2,:], label="t=1")
plot!(dset[3,:], label="t=2")
plot!(dset2[3,:], label="t=2")
plot!(dset[4,:], label="t=3")
plot!(dset2[4,:], label="t=3")
plot!(dset[11,:], label="t=10")
plot!(dset2[11,:], label="t=10")
plot!(dset[21,:], label="t=20")
plot!(dset2[21,:], label="t=20")
plot!(dset[51,:], label="t=50")
plot!(dset2[51,:], label="t=50")
plot!(dset[101,:], label="t=100")
plot!(dset2[101,:], label="t=100")

xlabel!("Spatial Position")
ylabel!("Temperature")
title!("Heat Distribution Over Time")
display(plot!)