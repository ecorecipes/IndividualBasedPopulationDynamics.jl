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
    offspring_count, sample_recruit

export Size, RNGResource, VitalRates, DomainResource, EvictionPolicy
export ibm_world, ibm_step!, ibm_run!
export population_size, traits, trait_histogram

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

end # module
