# API Reference

## Discrete-time IBM

```@docs
ibm_world
ibm_step!
ibm_run!
population_size
traits
trait_histogram
```

## Continuous-time IBM (measure-valued PDMP)

```@docs
CTVitalRates
ibm_world_ct
ibm_step_ct!
ibm_run_ct!
```

## Super-individuals

```@docs
Weight
ibm_world_super
ibm_step_super!
ibm_run_super!
total_count
n_particles
merge_by_bin!
```

## Stage structure (pure-jump finite-state IBM)

```@docs
Stage
StageVitalRates
ibm_world_stage
ibm_step_stage!
ibm_run_stage!
stage_counts
```

## Components and resources

```@docs
Size
RNGResource
VitalRates
DomainResource
EvictionPolicy
```

## SciML problem and integrator

An individual-based model as a SciML problem, so that `solve`, ensembles and
simulation-based resilience metrics (`recovery`, `basin_stability` and the
early-warning indicators of DynamicalResilience.jl) apply to it. The state is the
population binned on the mesh (or counted by stage).

```@docs
IBMProblem
IBMStepping
IBMIntegrator
IBMSolution
seed_trajectory
init(::IBMProblem)
solve(::IBMProblem)
remake(::IBMProblem)
reinit!(::IBMIntegrator)
set_u!(::IBMIntegrator, ::Any)
IndividualBasedPopulationDynamics.SciMLBase.u_modified!(::IBMIntegrator, ::Bool)
terminate!(::IBMIntegrator)
IndividualBasedPopulationDynamics.SciMLBase.check_error(::IBMIntegrator)
IndividualBasedPopulationDynamics.SciMLBase.successful_retcode(::IBMSolution)
IndividualBasedPopulationDynamics.SciMLBase.EnsembleProblem(::IBMProblem)
```
