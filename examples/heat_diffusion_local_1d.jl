include("heat_diffusion_common.jl")

function run_local_heat_1d(; n=128, alpha=1.0, cfl=0.4, domain_length=1.0, nt=100)
    u = LocalHaloArray(Float64, (n,), 1; boundary_condition=:periodic)
    dx = (domain_length / n,)
    dt = stable_heat_dt(alpha, cfl, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0, widths=(n / 12,))
    solve_heat!(u; alpha, dt, dx, nt)
    return u
end

if abspath(PROGRAM_FILE) == @__FILE__
    u = run_local_heat_1d()
    println("1D local heat diffusion completed. Final mean = ", sum(interior_view(u)) / length(u))
end
