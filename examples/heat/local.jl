include("common.jl")

# Local heat diffusion in 1-D, 2-D, and 3-D.  All three share the explicit
# stencil solver in common.jl; only the grid shape and Gaussian widths differ.
function run_local_heat(n; alpha=1.0, cfl=0.4,
        domain_length=ntuple(_ -> 1.0, length(n)), nt=100)
    u  = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    dx = ntuple(i -> domain_length[i] / n[i], length(n))
    dt = stable_heat_dt(alpha, cfl, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0,
        widths=ntuple(i -> n[i] / 12, length(n)))
    solve_heat!(u; alpha, dt, dx, nt)
    return u
end

if abspath(PROGRAM_FILE) == @__FILE__
    for (n, nt) in (((128,), 100), ((64, 64), 100), ((32, 32, 32), 50))
        u = run_local_heat(n; nt)
        println(length(n), "D local heat diffusion completed.  grid=", n,
            "  final mean = ", sum(interior_view(u)) / length(u))
    end
end
