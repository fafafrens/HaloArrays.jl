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
        "Examples" => "examples.md",
        "API reference" => [
            "Types" => "api/types.md",
            "Arrays, layout & reductions" => "api/core.md",
            "Halo exchange & boundary conditions" => "api/exchange.md",
            "Loops & kernel regions" => "api/loops.md",
        ],
    ],
    checkdocs = :none,   # the package has many internal helpers with docstrings
    warnonly = true,     # don't fail the build on missing @ref targets
)

deploydocs(
    repo = "github.com/fafafrens/HaloArrays.jl.git",
    devbranch = "main",
)
