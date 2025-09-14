using MPI
using HDF5
using HaloArrays


function heat_3d_step!( u_old::HaloArray, α, dt, dx)
    
    data_old=u_old.data

@inbounds for I in CartesianIndices(HaloArrays.interior_range(u_old))
        # Because of halos, interior indices shifted by h
            
e_i=HaloArrays.versors(u_old)
        laplacian = zero(eltype(data_old))

        @inbounds for dim in 1:(ndims(u_old))
        # Discrete Laplacian using halo data
        v_i=CartesianIndex(e_i[dim])
        I₊=I + v_i
        I₋=I - v_i

        laplacian = laplacian + (data_old[I₊] -2*data_old[I] + data_old[I₋]) / dx[dim]^2
        end
        data_old[I] =  data_old[I] + α * dt * laplacian
    end
    return nothing
end


function run_heat_3d_periodic(N_local, α, CFL, L, nt)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # 2D periodic domain setup
    dims = MPI.Dims_create(nproc, (0, 0, 0))
    periods = (true, true, true)  # periodic boundary in 2D
    topology=CartesianTopology(comm, Tuple(dims), periodic=periods)

    Nglobal = N_local .* dims

    dx = L ./ Nglobal  # spatial step size
    dt = CFL / (α * sum(1 ./ dx.^2))
    if rank == 0
    println("dx = $dx, dt = $dt, CFL actual = ", α * dt * sum(1 ./ dx.^2))
    end
    h = 1
    #u= HaloArray(Float64,N_local,h,topology; boundary_condition = (:repeating, :repeating))
    
    #u= HaloArray(Float64,N_local,h,topology; boundary_condition = (:repeating, :repeating))
    u= HaloArray(Float64,N_local,h,topology; boundary_condition = (:periodic, :periodic, :periodic))
    for r in 0:MPI.Comm_size(comm)-1
    MPI.Barrier(comm)
    if rank == r
        println("====== Rank $rank ======")
        println("Coords: ", topology.cart_coords)
        println("Neighbors: ", topology.neighbors)
        println("Neighbors 1 d : ", topology.neighbors[1])
        println("Coords Neighbors 1 d : ", MPI.Cart_coords.(Ref(topology.cart_comm), topology.neighbors[1]))
        println("Neighbors 2 d : ", topology.neighbors[2])
        println("Coords Neighbors 2 d : ", MPI.Cart_coords.(Ref(topology.cart_comm), topology.neighbors[2]))
        println()
    end
end
MPI.Barrier(comm)


    # Initialize Gaussian pulse across global domain

HaloArrays.fill_from_global_indices!(u) do i
    x = i[1]
    y = i[2]
    z= i[3]
    return  10 * exp(-(x - (Nglobal[1]/2))^2/5^2 - (y - (Nglobal[2]/2))^2/10^2 - (z - (Nglobal[3]/2))^2/15^2) + 1
    end
    
    gather_and_append_haloarray!("result","temp",u; root=0)
    append_haloarray_to_file!("result_par","temp",u)
    #fid ,dset=create_haloarray_output_file("result_par_fix.h5", "temp", u, nt+1)
    #write_haloarray_timestep!(dset, u, 1)  # Write initial condition

    MPI.Barrier(comm)
    for step in 1:nt
        # Halo exchange updates ghost cells, periodic neighbors wrap correctly
        halo_exchange_waitall_unsafe!(u)
        #this does nothing since everthing is worked  in the excange
        boundary_condition!(u)
        if any(isnan, u.data)
            @show step, u.data
            error("Trovato NaN dopo halo_exchange!")
        end

        # Compute heat equation step
        heat_3d_step!(u, α, dt, dx)

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




N_local=(100,100,100)
L = 1.0
CFL= 0.4  # CFL condition for stability

alpha = 1.0  # thermal diffusivity
nt = 10 # number of time steps
 
run_heat_3d_periodic(N_local, alpha, CFL, L, nt)

using PProf

@pprof run_heat_3d_periodic(N_local, alpha, CFL, L, nt)

#if false 
file = h5open("result.h5") 
file2 = h5open("result_par.h5") 

dset = file["temp"]
dset2 = file2["temp"]

size(dset)
size(dset2)

# Choose how many frames you want in the animation
frames = 1:nt  # or something like 1:5:nt for fewer frames
using Plots

heatmap(dset[1,:,:,50],
        title="Time step 1",
        xlabel="x",
        ylabel="y")

@gif for t in frames
    # Plot each time step
    a=dset[t,:,:,50]
    heatmap(a,
        title="Time step $t",
        xlabel="x",
        ylabel="y")
end every 1 

using Plots
plot(dset[1,:,100,50], label="t=0")
plot!(dset2[1,:,100,50], label="t=0")
plot!(dset[10,:,100,50], label="t=1")
plot!(dset2[10,:,100,50], label="t=1")




#end 



#plot!(dset[3,:], label="t=2")
#plot!(dset2[3,:], label="t=2")
#plot!(dset[4,:], label="t=3")
#plot!(dset2[4,:], label="t=3")
#plot!(dset[11,:], label="t=10")
#plot!(dset2[11,:], label="t=10")
#plot!(dset[21,:], label="t=20")
#plot!(dset2[21,:], label="t=20")
#plot!(dset[51,:], label="t=50")
#plot!(dset2[51,:], label="t=50")
#plot!(dset[101,:], label="t=100")
#plot!(dset2[101,:], label="t=100")#

#xlabel!("Spatial Position")
#ylabel!("Temperature")
#title!("Heat Distribution Over Time")
#display(plot!)