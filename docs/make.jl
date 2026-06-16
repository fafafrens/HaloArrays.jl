using Documenter
using HaloArrays

makedocs(
    sitename = "HaloArrays.jl",
    modules = [HaloArrays],
    authors = "Eduardo Grossi",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://fafafrens.github.io/HaloArrays.jl",
        size_threshold = 500_000,        # the single-page API reference is large
        size_threshold_warn = 400_000,
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Linear & implicit solves" => "solvers.md",
        "Examples" => "examples.md",
        "API" => [
            "Essentials" => "api/essentials.md",
            "Reference" => [
                "Types" => "api/types.md",
                "Arrays, layout & reductions" => "api/core.md",
                "Halo exchange & boundary conditions" => "api/exchange.md",
                "Loops & kernel regions" => "api/loops.md",
            ],
        ],
    ],
    checkdocs = :exports,           # every exported symbol must carry a docstring
    warnonly = [:cross_references], # missing @ref targets warn; everything else (incl. missing docs) fails
)

deploydocs(
    repo = "github.com/fafafrens/HaloArrays.jl.git",
    devbranch = "main",
)
