using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
if size != 4
    error("Please run with exactly 4 MPI processes")
end

dims = (2, 2)
periods = (false, false)
reorder = false
cart_comm = MPI.Cart_create(comm, dims, periods, reorder)
cart_coords = MPI.Cart_coords(cart_comm, rank)

local_data = [rank*10 + i + 4*j for j in 0:1, i in 1:2]

println("Rank $rank coords=$cart_coords local_data = $local_data")

# Local reduction along rows (dim=1)
dim = 1
local_reduced = sum(local_data, dims=dim)  # sum rows
println("Rank $rank local_reduced = $local_reduced")

# Create sub-communicator excluding reduced dimension
remain_dims = [i != dim for i in 1:2]
sub_comm = MPI.Cart_sub(cart_comm, remain_dims)
sub_rank = MPI.Comm_rank(sub_comm)
sub_size = MPI.Comm_size(sub_comm)

println("Rank $rank sub_comm rank = $sub_rank size = $sub_size")

# MPI reduce sum within sub_comm
sendbuf = vec(local_reduced)
recvbuf = similar(sendbuf)

root = 0
MPI.Reduce!(sendbuf, recvbuf, MPI.SUM, root, sub_comm)

if sub_rank == 0
    global_reduced = reshape(recvbuf, size(local_reduced))
    println("Rank $rank (root in sub_comm) global reduced = $global_reduced")
end

MPI.Barrier(comm)
MPI.Finalize()