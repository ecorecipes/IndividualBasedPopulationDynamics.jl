"""
    IndividualBasedPopulationDynamics

Individual-based (agent / ECS) realizations of structured population models, built
on the [Ark.jl](https://github.com/ark-ecs/Ark.jl) entity-component-system. Each
individual is an entity carrying a continuous `Size` trait; birth spawns an
entity, death despawns one, growth updates the trait in place. Vital-rate
*sampling* is delegated to the `StructuredPopulationCore` sampler interface
(`sample_survives`, `sample_growth`, `offspring_count`, `sample_recruit`), whose
analytic methods for the IPM vital-rate types live in `IntegralProjectionModels`.

This is the finite-N "ground truth" whose large-N limit is the deterministic IPM.
"""
module IndividualBasedPopulationDynamics

import Ark
import Random
using StructuredPopulationCore: sample_survives, sample_growth,
    offspring_count, expected_offspring, sample_recruit, rand_poisson, rand_binomial

export Size, RNGResource, VitalRates, DomainResource, EvictionPolicy
export ibm_world, ibm_step!, ibm_run!
export population_size, traits, trait_histogram
export CTVitalRates, ibm_world_ct, ibm_step_ct!, ibm_run_ct!
export Weight, ibm_world_super, ibm_step_super!, ibm_run_super!
export total_count, n_particles, merge_by_bin!
export Stage, StageVitalRates, ibm_world_stage, ibm_step_stage!, ibm_run_stage!, stage_counts

# ---------------------------------------------------------------------------
# Components & resources
# ---------------------------------------------------------------------------

"""Individual's continuous trait (e.g. size). Immutable — updated by write-back."""
struct Size
    z::Float64
end

# Resources are keyed by their (concrete, non-parametric) type in Ark's resource
# store, so these stay non-parametric; `ibm_step!` re-types the fields through a
# function barrier to keep the per-individual loop type-stable.

"""Resource holding the simulation RNG."""
struct RNGResource
    rng::Random.AbstractRNG
end

"""Resource holding the vital-rate objects (survival, growth, fecundity)."""
struct VitalRates
    survival::Any
    growth::Any
    fecundity::Any
end

"""Resource holding the trait domain (for eviction and the density-sampler fallback)."""
struct DomainResource
    domain::Any
end

"""Resource selecting the out-of-domain policy: `:reflect`, `:clamp`, or `:discard`."""
struct EvictionPolicy
    mode::Symbol
end

"""Super-individual weight: the number of real individuals a particle represents."""
struct Weight
    w::Int
end

# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

# Return an in-domain trait, or NaN to signal "evicted out" (treated as mortality).
function _evict(z, domain, mode::Symbol)
    lo = domain.lower
    hi = domain.upper
    (lo <= z <= hi) && return z
    mode === :clamp && return clamp(z, lo, hi)
    if mode === :reflect
        width = hi - lo
        y = mod(z - lo, 2width)
        return lo + (y <= width ? y : 2width - y)
    end
    return NaN   # :discard
end

# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

"""
    ibm_world(survival, growth, fecundity, domain; rng, traits0, eviction=:reflect)

Build an Ark `World` of `Size` entities for an individual-based realization, with
the vital rates, RNG, domain, and eviction policy stored as resources and one
entity spawned per value in `traits0`.
"""
function ibm_world(survival, growth, fecundity, domain;
        rng::Random.AbstractRNG = Random.default_rng(),
        traits0::AbstractVector = Float64[],
        eviction::Symbol = :reflect)
    world = Ark.World(Size)
    Ark.add_resource!(world, RNGResource(rng))
    Ark.add_resource!(world, VitalRates(survival, growth, fecundity))
    Ark.add_resource!(world, DomainResource(domain))
    Ark.add_resource!(world, EvictionPolicy(eviction))
    for z in traits0
        Ark.new_entity!(world, (Size(float(z)),))
    end
    return world
end

# ---------------------------------------------------------------------------
# Discrete-time IPM step
# ---------------------------------------------------------------------------

"""
    ibm_step!(world)

Advance one discrete time step. Within a single `Query` (which locks the world)
each individual's offspring and survival/growth are resolved against its *current*
trait; structural changes (deaths, births) are buffered and applied after the
query completes. The conditional mean reproduces the deterministic IPM kernel
`K = P + F`.
"""
function ibm_step!(world)
    vr = Ark.get_resource(world, VitalRates)
    rng = Ark.get_resource(world, RNGResource).rng
    domain = Ark.get_resource(world, DomainResource).domain
    mode = Ark.get_resource(world, EvictionPolicy).mode
    # function barrier: concrete vital-rate types -> type-stable inner loop
    return _ibm_step!(world, rng, vr.survival, vr.growth, vr.fecundity, domain, mode)
end

function _ibm_step!(world, rng, survival, growth, fecundity, domain, mode::Symbol)
    dead = Ark.Entity[]
    offspring = Float64[]
    for q in Ark.Query(world, (Size,))
        eids, sizes = q
        for i in eachindex(eids)
            z = sizes[i].z
            # reproduction from the current (parent) trait
            for _ in 1:offspring_count(rng, fecundity, z, domain)
                zr = _evict(sample_recruit(rng, fecundity, z, domain), domain, mode)
                isnan(zr) || push!(offspring, zr)
            end
            # survival + growth (in-place write-back), else death
            if sample_survives(rng, survival, z)
                zg = _evict(sample_growth(rng, growth, z, domain), domain, mode)
                if isnan(zg)
                    push!(dead, eids[i])      # evicted-out survivor counts as mortality
                else
                    sizes[i] = Size(zg)
                end
            else
                push!(dead, eids[i])
            end
        end
    end
    for e in dead
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    for z in offspring
        Ark.new_entity!(world, (Size(z),))
    end
    return world
end

# ---------------------------------------------------------------------------
# Driver & summaries
# ---------------------------------------------------------------------------

"""
    population_size(world) -> Int

Number of living individuals. Counts via a completed `Query` iteration (which
releases the world lock — `count_entities(Query(...))` would leave it locked).
"""
function population_size(world)
    n = 0
    for q in Ark.Query(world, (Size,))
        eids, _ = q
        n += length(eids)
    end
    return n
end

"""
    traits(world) -> Vector{Float64}

Trait values of all living individuals.
"""
function traits(world)
    out = Float64[]
    for q in Ark.Query(world, (Size,))
        _, sizes = q
        for i in eachindex(sizes)
            push!(out, sizes[i].z)
        end
    end
    return out
end

"""
    ibm_run!(world, n_steps; save_traits=false) -> (; N, traits)

Run `n_steps` discrete steps in place. Returns the population-size trajectory `N`
(length `n_steps+1`, including the initial size) and, if `save_traits`, the trait
vectors at each saved time.
"""
function ibm_run!(world, n_steps::Int; save_traits::Bool = false)
    N = Int[population_size(world)]
    snaps = Vector{Float64}[]
    save_traits && push!(snaps, traits(world))
    for _ in 1:n_steps
        ibm_step!(world)
        push!(N, population_size(world))
        save_traits && push!(snaps, traits(world))
    end
    return (N = N, traits = snaps)
end

# ---------------------------------------------------------------------------
# Continuous-time IBM (measure-valued PDMP): deterministic flow + jumps
# ---------------------------------------------------------------------------

"""Per-individual continuous-time vital rates: deterministic growth `flow(x)`,
`mortality(x)` death rate, `fecundity(x)` birth rate, and `recruit(rng)` returning
an offspring trait. (Non-parametric for Ark resource keying.)"""
struct CTVitalRates
    flow::Any
    mortality::Any
    fecundity::Any
    recruit::Any
end

"""
    ibm_world_ct(flow, mortality, fecundity, recruit, domain; rng, traits0, eviction=:reflect)

Build an Ark `World` for a continuous-time individual-based realization (a
measure-valued PDMP): individuals flow deterministically by `flow(x)` and undergo
death (rate `mortality(x)`) and birth (rate `fecundity(x)`, offspring trait from
`recruit(rng)`) jumps. This is the individual-level dual of a PSPM transport
problem (`flow`=velocity, `mortality`=mortality, births=boundary/fecundity).
"""
function ibm_world_ct(flow, mortality, fecundity, recruit, domain;
        rng::Random.AbstractRNG = Random.default_rng(),
        traits0::AbstractVector = Float64[],
        eviction::Symbol = :reflect)
    world = Ark.World(Size)
    Ark.add_resource!(world, RNGResource(rng))
    Ark.add_resource!(world, CTVitalRates(flow, mortality, fecundity, recruit))
    Ark.add_resource!(world, DomainResource(domain))
    Ark.add_resource!(world, EvictionPolicy(eviction))
    for z in traits0
        Ark.new_entity!(world, (Size(float(z)),))
    end
    return world
end

"""
    ibm_step_ct!(world, dt)

Advance continuous time by `dt` with an operator split: over `[t, t+dt]` each
individual produces `Poisson(fecundity(x)·dt)` offspring, dies with probability
`1 - exp(-mortality(x)·dt)`, and otherwise flows to `x + flow(x)·dt` (Euler).
Exact as `dt → 0`; the mean obeys the transport/​generator equation.
"""
function ibm_step_ct!(world, dt)
    vr = Ark.get_resource(world, CTVitalRates)
    rng = Ark.get_resource(world, RNGResource).rng
    domain = Ark.get_resource(world, DomainResource).domain
    mode = Ark.get_resource(world, EvictionPolicy).mode
    return _ibm_step_ct!(world, float(dt), rng, vr.flow, vr.mortality, vr.fecundity,
        vr.recruit, domain, mode)
end

function _ibm_step_ct!(world, dt, rng, flow, mortality, fecundity, recruit, domain, mode::Symbol)
    dead = Ark.Entity[]
    offspring = Float64[]
    for q in Ark.Query(world, (Size,))
        eids, sizes = q
        for i in eachindex(eids)
            x = sizes[i].z
            for _ in 1:rand_poisson(rng, fecundity(x) * dt)
                zr = _evict(recruit(rng), domain, mode)
                isnan(zr) || push!(offspring, zr)
            end
            if rand(rng) < -expm1(-mortality(x) * dt)        # 1 - exp(-μ dt)
                push!(dead, eids[i])
            else
                xn = _evict(x + flow(x) * dt, domain, mode)
                isnan(xn) ? push!(dead, eids[i]) : (sizes[i] = Size(xn))
            end
        end
    end
    for e in dead
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    for z in offspring
        Ark.new_entity!(world, (Size(z),))
    end
    return world
end

"""
    ibm_run_ct!(world, tspan; dt, saveat=dt, save_traits=false) -> (; t, N, traits)

Run the continuous-time IBM over `tspan` with step `dt`, recording population size
(and optionally traits) every `saveat`.
"""
function ibm_run_ct!(world, tspan; dt::Real, saveat::Real = dt, save_traits::Bool = false)
    t0 = float(tspan[1])
    tf = float(tspan[2])
    nsteps = round(Int, (tf - t0) / dt)
    stride = max(1, round(Int, saveat / dt))
    ts = Float64[t0]
    N = Int[population_size(world)]
    snaps = Vector{Float64}[]
    save_traits && push!(snaps, traits(world))
    for k in 1:nsteps
        ibm_step_ct!(world, dt)
        if k % stride == 0 || k == nsteps
            push!(ts, t0 + k * dt)
            push!(N, population_size(world))
            save_traits && push!(snaps, traits(world))
        end
    end
    return (t = ts, N = N, traits = snaps)
end

"""
    trait_histogram(world, domain) -> Vector{Int}

Counts of living individuals per mesh bin of `domain` (for comparison with the
deterministic density).
"""
function trait_histogram(world, domain)
    n = domain.n_meshpoints
    lo = domain.lower
    h = (domain.upper - domain.lower) / n
    counts = zeros(Int, n)
    for q in Ark.Query(world, (Size,))
        _, sizes = q
        for i in eachindex(sizes)
            b = clamp(floor(Int, (sizes[i].z - lo) / h) + 1, 1, n)
            counts[b] += 1
        end
    end
    return counts
end

# ---------------------------------------------------------------------------
# Super-individuals: weighted particles representing large true-N with few entities
# ---------------------------------------------------------------------------

"""
    ibm_world_super(survival, growth, fecundity, domain; rng, traits0, weights0, eviction=:reflect)

Build an Ark `World` of weighted super-individuals (`Size` + `Weight`): each
particle represents `Weight.w` real individuals at one trait. `weights0` defaults
to all-ones (the exact IBM). Demographic events use exact aggregate draws
(`Binomial` survivors, `Poisson` offspring), so total-count statistics match the
unweighted IBM while the entity count stays small.
"""
function ibm_world_super(survival, growth, fecundity, domain;
        rng::Random.AbstractRNG = Random.default_rng(),
        traits0::AbstractVector = Float64[],
        weights0::AbstractVector{<:Integer} = ones(Int, length(traits0)),
        eviction::Symbol = :reflect)
    length(weights0) == length(traits0) || throw(DimensionMismatch(
        "weights0 and traits0 must have equal length"))
    world = Ark.World(Size, Weight)
    Ark.add_resource!(world, RNGResource(rng))
    Ark.add_resource!(world, VitalRates(survival, growth, fecundity))
    Ark.add_resource!(world, DomainResource(domain))
    Ark.add_resource!(world, EvictionPolicy(eviction))
    for (z, w) in zip(traits0, weights0)
        w > 0 && Ark.new_entity!(world, (Size(float(z)), Weight(Int(w))))
    end
    return world
end

"""
    ibm_step_super!(world)

One discrete step for weighted super-individuals. A particle `(z, w)` draws its
aggregate offspring/survivor counts with `Poisson(w · expected_offspring(z))`
and `Binomial(w, survival(z))`, but each represented individual then gets an
independent recruit/growth trait draw. This preserves the super-individual
count-process statistics while restoring within-particle trait variance.
"""
function ibm_step_super!(world)
    vr = Ark.get_resource(world, VitalRates)
    rng = Ark.get_resource(world, RNGResource).rng
    domain = Ark.get_resource(world, DomainResource).domain
    mode = Ark.get_resource(world, EvictionPolicy).mode
    return _ibm_step_super!(world, rng, vr.survival, vr.growth, vr.fecundity, domain, mode)
end

function _ibm_step_super!(world, rng, survival, growth, fecundity, domain, mode::Symbol)
    dead = Ark.Entity[]
    offspring = Float64[]
    survivors = Float64[]
    for q in Ark.Query(world, (Size, Weight))
        eids, sizes, weights = q
        for i in eachindex(eids)
            z = sizes[i].z
            w = weights[i].w
            k = rand_poisson(rng, w * expected_offspring(fecundity, z, domain))
            for _ in 1:k
                zr = _evict(sample_recruit(rng, fecundity, z, domain), domain, mode)
                isnan(zr) || push!(offspring, zr)
            end
            b = rand_binomial(rng, w, clamp(survival(z), 0.0, 1.0))
            alive = 0
            for _ in 1:b
                zg = _evict(sample_growth(rng, growth, z, domain), domain, mode)
                if !isnan(zg)
                    alive += 1
                    if alive == 1
                        sizes[i] = Size(zg)
                        weights[i] = Weight(1)
                    else
                        push!(survivors, zg)
                    end
                end
            end
            alive == 0 && push!(dead, eids[i])
        end
    end
    for e in dead
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    for z in survivors
        Ark.new_entity!(world, (Size(z), Weight(1)))
    end
    for z in offspring
        Ark.new_entity!(world, (Size(z), Weight(1)))
    end
    return world
end

"""
    total_count(world) -> Int

Total number of real individuals (sum of super-individual weights).
"""
function total_count(world)
    s = 0
    for q in Ark.Query(world, (Weight,))
        _, ws = q
        for i in eachindex(ws)
            s += ws[i].w
        end
    end
    return s
end

"""
    n_particles(world) -> Int

Number of super-individual particles (entities). `total_count(world)` is the
represented population.
"""
n_particles(world) = population_size(world)

"""
    merge_by_bin!(world, domain) -> world

Cap particle count by merging all weight in each `domain` mesh bin into a single
particle at the bin midpoint (total weight conserved; trait resolution = mesh).
"""
function merge_by_bin!(world, domain)
    n = domain.n_meshpoints
    lo = domain.lower
    h = (domain.upper - domain.lower) / n
    binw = zeros(Int, n)
    ents = Ark.Entity[]
    for q in Ark.Query(world, (Size, Weight))
        eids, sizes, weights = q
        for i in eachindex(eids)
            b = clamp(floor(Int, (sizes[i].z - lo) / h) + 1, 1, n)
            binw[b] += weights[i].w
            push!(ents, eids[i])
        end
    end
    for e in ents
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    for b in 1:n
        binw[b] > 0 && Ark.new_entity!(world, (Size(lo + (b - 0.5) * h), Weight(binw[b])))
    end
    return world
end

"""
    ibm_run_super!(world, n_steps; merge_domain=nothing) -> (; N, particles)

Run `n_steps` weighted steps, returning the total-count trajectory `N` and the
particle-count trajectory. If `merge_domain` is a domain, `merge_by_bin!` is
applied after each step to cap the particle count.
"""
function ibm_run_super!(world, n_steps::Int; merge_domain = nothing)
    N = Int[total_count(world)]
    particles = Int[n_particles(world)]
    for _ in 1:n_steps
        ibm_step_super!(world)
        merge_domain === nothing || merge_by_bin!(world, merge_domain)
        push!(N, total_count(world))
        push!(particles, n_particles(world))
    end
    return (N = N, particles = particles)
end

# ---------------------------------------------------------------------------
# Stage-structured (pure-jump finite-state) continuous-time IBM
# ---------------------------------------------------------------------------

"""Discrete stage of an individual (the individual-based analogue of a finite
state). Multi-component individuals combine `Stage` with `Size`/`Age`."""
struct Stage
    s::Int
end

"""Per-stage continuous-time rates: `Qtrans[s', s]` = rate of an `s`-individual
transitioning to `s'` (off-diagonal movement), `death[s]` mortality rate, and
birth matrix `birth[s', s]` = rate an `s`-individual produces an offspring in
stage `s'`. (Non-parametric for Ark resource keying.)"""
struct StageVitalRates
    Qtrans::Any
    death::Any
    birth::Any
end

"""
    ibm_world_stage(Qtrans, death, birth; rng, stages0)

Build an Ark `World` of `Stage` individuals for a stage-structured continuous-time
IBM — a pure-jump Markov process whose mean is the finite-state generator
`dn/dt = G·n` (with `G` assembled from the transition/death/birth rates; `birth`
is the offspring matrix `birth[to, from]`). One entity is spawned per entry of
`stages0`.
"""
function ibm_world_stage(Qtrans, death, birth;
        rng::Random.AbstractRNG = Random.default_rng(),
        stages0::AbstractVector{<:Integer} = Int[])
    world = Ark.World(Stage)
    Ark.add_resource!(world, RNGResource(rng))
    Ark.add_resource!(world, StageVitalRates(Qtrans, death, birth))
    for s in stages0
        Ark.new_entity!(world, (Stage(Int(s)),))
    end
    return world
end

"""
    ibm_step_stage!(world, dt)

One operator-split step: each individual in stage `s` produces
`Poisson(birth[to, s]·dt)` offspring in each stage `to`, and with probability
`1 - exp(-R·dt)` (`R` = total transition + death rate) undergoes one event —
death, or a transition to `s'` with probability `Qtrans[s', s] / R` (an in-place
stage change).
"""
function ibm_step_stage!(world, dt)
    vr = Ark.get_resource(world, StageVitalRates)
    rng = Ark.get_resource(world, RNGResource).rng
    n = size(vr.Qtrans, 1)
    return _ibm_step_stage!(world, float(dt), rng, vr.Qtrans, vr.death, vr.birth, n)
end

function _ibm_step_stage!(world, dt, rng, Qtrans, death, birth, n::Int)
    dead = Ark.Entity[]
    offspring = Int[]
    for q in Ark.Query(world, (Stage,))
        eids, stages = q
        for i in eachindex(eids)
            s = stages[i].s
            @inbounds for to in 1:n
                r = birth[to, s]
                if r > 0
                    for _ in 1:rand_poisson(rng, r * dt)
                        push!(offspring, to)
                    end
                end
            end
            rate_out = 0.0
            @inbounds for sp in 1:n
                sp != s && (rate_out += Qtrans[sp, s])
            end
            R = rate_out + death[s]
            if R > 0 && rand(rng) < -expm1(-R * dt)
                u = rand(rng) * R
                if u < death[s]
                    push!(dead, eids[i])
                else
                    u -= death[s]
                    cum = 0.0
                    target = s
                    @inbounds for sp in 1:n
                        sp == s && continue
                        cum += Qtrans[sp, s]
                        if u <= cum
                            target = sp
                            break
                        end
                    end
                    stages[i] = Stage(target)
                end
            end
        end
    end
    for e in dead
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    for sg in offspring
        Ark.new_entity!(world, (Stage(sg),))
    end
    return world
end

"""
    stage_counts(world, n_stages) -> Vector{Int}

Number of individuals in each stage `1:n_stages`.
"""
function stage_counts(world, n_stages::Int)
    c = zeros(Int, n_stages)
    for q in Ark.Query(world, (Stage,))
        _, stages = q
        for i in eachindex(stages)
            c[stages[i].s] += 1
        end
    end
    return c
end

"""
    ibm_run_stage!(world, tspan; dt, saveat=dt, n_stages) -> (; t, counts)

Run the stage-structured IBM over `tspan` with step `dt`, recording the per-stage
counts every `saveat`.
"""
function ibm_run_stage!(world, tspan; dt::Real, saveat::Real = dt, n_stages::Int)
    t0 = float(tspan[1])
    tf = float(tspan[2])
    nsteps = round(Int, (tf - t0) / dt)
    stride = max(1, round(Int, saveat / dt))
    ts = Float64[t0]
    counts = Vector{Int}[stage_counts(world, n_stages)]
    for k in 1:nsteps
        ibm_step_stage!(world, dt)
        if k % stride == 0 || k == nsteps
            push!(ts, t0 + k * dt)
            push!(counts, stage_counts(world, n_stages))
        end
    end
    return (t = ts, counts = counts)
end

end # module
