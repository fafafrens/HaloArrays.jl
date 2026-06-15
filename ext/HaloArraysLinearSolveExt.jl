module HaloArraysLinearSolveExt

using HaloArrays
using LinearSolve
using Krylov

# Krylov methods usable on a halo array: LinearSolve's KrylovJL machinery wraps
# them, and they allocate all their work vectors up front (so they never hit the
# `S(undef, n)` path mid-solve — see the docstring). The user-facing symbol maps
# to Krylov's in-place solver `Krylov.<method>!`.
const _HALOKRYLOV_METHODS = (:gmres, :cg, :bicgstab, :minres, :dqgmres, :diom,
                             :fom, :cgs, :minares, :minres_qlp, :symmlq)

function HaloArrays.HaloKrylov(method::Symbol; kwargs...)
    method in _HALOKRYLOV_METHODS || throw(ArgumentError(string(
        "HaloKrylov: unsupported method :", method, ". Supported (matrix-free, ",
        "cached via LinearSolve's KrylovJL): ", join(_HALOKRYLOV_METHODS, ", "), ". ",
        ":fgmres/:qmr/:bilq allocate via S(undef, n) mid-solve; :car/:cg_lanczos ",
        "are not wrapped by LinearSolve's KrylovJL.")))
    return KrylovJL(; KrylovAlg = getproperty(Krylov, Symbol(method, :!)), kwargs...)
end

# Teach LinearSolve's KrylovJL to build its Krylov workspace via `KrylovConstructor`
# (i.e. `similar(b)`) when the solver vector is a halo array. The stock path
# allocates work vectors with `S(undef, n)`, which a geometry-carrying halo array
# has no constructor for; KrylovConstructor sidesteps that. This mirrors
# LinearSolve's own generic `init_cacheval` (and its `ArrayPartition`
# specialization), swapping `KS(A, b)` → `KS(KrylovConstructor(b))`. LinearSolve's
# `solve!` stores the returned workspace in `cache.cacheval`, so it is built once
# and reused across solves — no per-solve workspace allocation.
function LinearSolve.init_cacheval(alg::KrylovJL, A, b::AbstractHaloArray, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearSolve.LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions; zeroinit = true)
    KS = LinearSolve.get_KrylovJL_solver(alg.KrylovAlg)
    kc = Krylov.KrylovConstructor(b)
    memory_methods = (Krylov.dqgmres!, Krylov.diom!, Krylov.gmres!, Krylov.fgmres!,
                      Krylov.gpmr!, Krylov.fom!)
    solver = if zeroinit
        alg.KrylovAlg in memory_methods ? KS(kc; memory = 1) : KS(kc)
    else
        knt = NamedTuple(alg.kwargs)
        memory = haskey(knt, :memory) ? knt[:memory] :
                 (alg.gmres_restart == 0 ? min(20, size(A, 1)) : alg.gmres_restart)
        if alg.KrylovAlg in memory_methods
            KS(kc; memory)
        elseif alg.KrylovAlg in (Krylov.minres!, Krylov.symmlq!, Krylov.lslq!,
                                 Krylov.lsqr!, Krylov.lsmr!)
            alg.window != 0 ? KS(kc; window = alg.window) : KS(kc)
        else
            KS(kc)
        end
    end
    solver.x = u
    return solver
end

end # module
