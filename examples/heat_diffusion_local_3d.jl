include("heat_diffusion_common.jl")

function run_local_heat_3d(; n=(32, 32, 32), alpha=1.0, cfl=0.4, domain_length=(1.0, 1.0, 1.0), nt=50)
    u = LocalHaloArray(Float64, n, 1; boundary_condition=:periodic)
    dx = ntuple(i -> domain_length[i] / n[i], Val(3))
    dt = stable_heat_dt(alpha, cfl, dx)

    fill_centered_gaussian!(u; baseline=1.0, amplitude=1.0, widths=(n[1] / 12, n[2] / 12, n[3] / 12))
    solve_heat!(u; alpha, dt, dx, nt)
    return u
end

if abspath(PROGRAM_FILE) == @__FILE__
    u = run_local_heat_3d()
    println("3D local heat diffusion completed. Final mean = ", sum(interior_view(u)) / length(u))
end
