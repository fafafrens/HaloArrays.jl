using HDF5
using Plots

filename = "result.h5"
dataset = "temp"

ff = h5open(filename, "r")
arr = read(ff[dataset])
close(ff)

# Asse spaziale (r)
N = size(arr, 2)
L = 1.0  # Modifica se il dominio ha lunghezza diversa
r = range(0, L, length=N)

# Scegli i time step che vuoi plottare
timesteps = [1, 100, 500, 1000, 2000, 3000]  # Modifica a piacere

plt = plot()
for t in timesteps
    plot!(plt, r, arr[t, :], label="step $t")
end

xlabel!("r")
ylabel!("u(r)")
title!("Profili spaziali a diversi time step")
display(plt)