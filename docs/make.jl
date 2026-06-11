using Documenter
using IndividualBasedPopulationDynamics

# Headless plotting for @example blocks (GR).
ENV["GKSwstype"] = "100"

makedocs(;
    modules = [IndividualBasedPopulationDynamics],
    warnonly = true,
    authors = "Simon Frost",
    sitename = "IndividualBasedPopulationDynamics.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/IndividualBasedPopulationDynamics.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Introduction" => "tutorials/01_introduction.md",
            "Continuous-time dynamics" => "tutorials/02_continuous_time.md",
            "Super-individuals & stages" => "tutorials/03_super_and_stage.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/IndividualBasedPopulationDynamics.jl.git",
    push_preview = true,
)
