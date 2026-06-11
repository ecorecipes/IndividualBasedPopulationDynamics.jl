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
