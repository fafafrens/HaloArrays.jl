using Test
using MPI
using HaloArrays

module LinearAdvectionDiffEqExample
include(joinpath(@__DIR__, "..", "examples", "linear_advection_diffeq_1d.jl"))
end

const LADE = LinearAdvectionDiffEqExample

@testset "Linear advection DiffEq analytic solution" begin
    nx = 128
    velocity = 1.0
    steps = 40
    cfl = 0.2
    err_tol = 2.5e-2
    mass_tol = 1.0e-12

    local_u, _, _, local_mass0, local_mass1, local_err =
        LADE.run_local_advection_diffeq(; nx, velocity, steps, cfl)
    @test local_u isa LocalHaloArray
    @test abs(local_mass1 - local_mass0) < mass_tol
    @test local_err < err_tol

    threaded_u, _, _, threaded_mass0, threaded_mass1, threaded_err =
        LADE.run_threaded_advection_diffeq(; nx, tile_dims=(4,), velocity, steps, cfl)
    @test threaded_u isa ThreadedHaloArray
    @test abs(threaded_mass1 - threaded_mass0) < mass_tol
    @test threaded_err < err_tol

    mpi_u, _, _, mpi_mass0, mpi_mass1, mpi_err =
        LADE.run_mpi_advection_diffeq(; nx, velocity, steps, cfl)
    @test mpi_u isa HaloArray
    @test abs(mpi_mass1 - mpi_mass0) < mass_tol
    @test mpi_err < err_tol

    @test abs(local_err - mpi_err) < mass_tol
end
