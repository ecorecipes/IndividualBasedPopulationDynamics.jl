# IndividualBasedPopulationDynamics.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/IndividualBasedPopulationDynamics.jl/dev/)

Individual-based (agent / ECS) realizations of structured population models in
Julia, built on the [Ark.jl](https://github.com/ark-ecs/Ark.jl)
entity-component-system. Each individual is an entity carrying a continuous
`Size` trait (and optionally a discrete `Stage` or a super-individual `Weight`);
birth spawns an entity, death despawns one, and growth updates the trait.

This is the finite-``N`` "ground truth" whose large-``N`` limit is the
deterministic model — so it reproduces the integral/​matrix/​generator dynamics in
the mean while additionally capturing demographic stochasticity, small-population
effects, and within-population trait variation. Vital-rate *sampling* is delegated
to the [StructuredPopulationCore.jl](https://github.com/ecorecipes/ProjectionModels.jl)
sampler interface, whose analytic methods for the
[IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl)
vital-rate types load automatically.

## Features

| Mode | Builder / stepper | Deterministic limit |
|------|-------------------|---------------------|
| Discrete-time, size-structured | `ibm_world` / `ibm_step!` / `ibm_run!` | the IPM ``n_{t+1} = K n_t`` |
| Continuous-time flow + jumps | `ibm_world_ct` / `ibm_step_ct!` / `ibm_run_ct!` | the PSPM transport / measure-valued PDMP |
| Super-individuals (weighted) | `ibm_world_super` / `ibm_step_super!` / `ibm_run_super!` | the same, with exact aggregate counts |
| Discrete-stage pure jumps | `ibm_world_stage` / `ibm_step_stage!` / `ibm_run_stage!` | the finite-state generator ``\dot n = G n`` |

- **Super-individuals**: a `Weight` per particle represents many real individuals;
  exact `Binomial`/`Poisson` aggregate draws keep total-count statistics identical
  to the full model while bounding the entity count (`merge_by_bin!`).
- **Eviction policies** (`:reflect`, `:clamp`, `:discard`) keep traits in-domain.
- **Categorical lowering target**:
  [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  ships a weakdep extension that lowers a stage-structured `ValuedProjectionNet` to
  an individual-based realization via `IBMStageTarget`.

## Installation

This package is not yet registered in the Julia General registry. Install directly
from GitHub (Ark.jl is also unregistered):

```julia
using Pkg
Pkg.add(url = "https://github.com/ark-ecs/Ark.jl")
Pkg.add(url = "https://github.com/ecorecipes/IndividualBasedPopulationDynamics.jl")
```

## Quick start

```julia
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

See the [documentation](https://ecorecipes.github.io/IndividualBasedPopulationDynamics.jl/dev/)
for tutorials (introduction, continuous-time dynamics, super-individuals & stages)
and the full API reference.

## Related

- [StructuredPopulationCore.jl](https://github.com/ecorecipes/ProjectionModels.jl)
  — shared abstractions and the vital-rate sampler interface
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl)
  — continuous-state, discrete-time integral projection models (the deterministic
  limit of the discrete-time IBM)
- [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl)
  — discrete-stage, discrete-time matrix projection models
- [FiniteStatePopulationDynamics.jl](https://github.com/ecorecipes/FiniteStatePopulationDynamics.jl)
  — finite-state continuous-time dynamics (the limit of the stage IBM)
- [ContinuousStatePopulationDynamics.jl](https://github.com/ecorecipes/ContinuousStatePopulationDynamics.jl)
  — continuous-state continuous-time generators / PSPM transport (the limit of the
  continuous-time IBM)
- [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  — compositional categorical front-end that lowers to this backend via
  `IBMStageTarget`
