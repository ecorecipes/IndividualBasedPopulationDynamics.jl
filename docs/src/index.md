# IndividualBasedPopulationDynamics.jl

Individual-based (agent / ECS) realizations of structured population models, built
on the [Ark.jl](https://github.com/ark-ecs/Ark.jl) entity-component-system.

Each individual is an entity carrying a continuous `Size` trait (and optionally a
discrete `Stage` or a super-individual `Weight`); birth spawns an entity, death
despawns one, and growth updates the trait. This is the finite-``N`` "ground
truth" whose large-``N`` limit is the deterministic model:

| Mode | Realization | Deterministic limit |
|------|-------------|---------------------|
| `ibm_world` / `ibm_step!` | discrete-time, size-structured | the IPM ``n_{t+1} = K n_t`` |
| `ibm_world_ct` / `ibm_step_ct!` | continuous-time flow + jumps | the PSPM transport / measure-valued PDMP |
| `ibm_world_super` / `ibm_step_super!` | weighted super-individuals | the same, with exact aggregate counts |
| `ibm_world_stage` / `ibm_step_stage!` | discrete-stage pure jumps | the finite-state generator ``\dot n = G n`` |

Vital-rate *sampling* is delegated to the
[`StructuredPopulationCore`](https://github.com/ecorecipes/StructuredPopulationCore.jl)
sampler interface, whose analytic methods for the
[`IntegralProjectionModels.jl`](https://github.com/ecorecipes/IntegralProjectionModels.jl)
vital-rate types are loaded automatically. A categorical specification can also be
lowered to an individual-based realization via `IBMStageTarget` in
[`CategoricalPopulationDynamics.jl`](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ark-ecs/Ark.jl")
Pkg.add(url = "https://github.com/ecorecipes/IndividualBasedPopulationDynamics.jl")
```

## Quick start

```@example quickstart
using IndividualBasedPopulationDynamics
using IntegralProjectionModels
using Random

survival  = LinearSurvival(1.0, 0.0)
growth    = NormalGrowth(0.5, 0.8, 0.4)
fecundity = FecundityRate(-1.0, 0.0, 1.0, 0.3, 1.0)
domain    = ContinuousDomain(0.0, 6.0, 60)

world = ibm_world(survival, growth, fecundity, domain;
                  rng = Random.Xoshiro(1), traits0 = fill(2.0, 1000))
ibm_run!(world, 5).N
```

See the [tutorials](tutorials/01_introduction.md) for worked examples and the
[API reference](api.md) for the full interface.
