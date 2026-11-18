# SciML's problem/integrator interface for the individual-based worlds, so that tools written
# against it (solve, ensembles, and the simulation-based resilience metrics of
# DynamicalResilience.jl: recovery after a pulse, basin stability, early warnings) apply to
# individual-based models. The state an integrator exposes is the population *binned on a
# mesh* (or counted by stage): the representation of the deterministic model the IBM
# converges to, so that IBM and IPM/generator states can be compared directly.

"""
    IBMStepping()

The algorithm of an [`IBMProblem`](@ref): advance the individual-based world one step at a
time with the package's own stepping functions ([`ibm_step!`](@ref),
[`ibm_step_ct!`](@ref) or [`ibm_step_stage!`](@ref)).
"""
struct IBMStepping <: SciMLBase.AbstractSciMLAlgorithm end

"""
    IBMProblem(rates, u0, tspan; domain=nothing, dt=1, seed=0, eviction=:reflect)

An individual-based population model as a SciML problem. `rates` chooses the model:

- [`VitalRates`](@ref)`(survival, growth, fecundity)`: the discrete-time size-structured IBM
  of [`ibm_world`](@ref) (one step per generation; `dt` must be 1);
- [`CTVitalRates`](@ref)`(flow, mortality, fecundity, recruit)`: the continuous-time IBM of
  [`ibm_world_ct`](@ref), stepped with `dt`;
- [`StageVitalRates`](@ref)`(Qtrans, death, birth)`: the stage-structured IBM of
  [`ibm_world_stage`](@ref), stepped with `dt`.

**State.** The state `u` is a `Vector{Float64}`: the number of individuals in each of the
`domain.n_meshpoints` equal bins of `domain` (`[lower, upper]`, as in
[`trait_histogram`](@ref)) for size-structured models, or in each stage for stage models.
`u0` is such a vector of counts; individuals are created from it with traits uniform within
their bin, using the run's random number generator, so a run is a function of `u0` and
`seed` only. Non-integer counts are rounded stochastically (to the floor or the ceiling,
with probabilities that keep the expected count).

**Randomness.** Each run uses `Random.Xoshiro(seed)`; `reinit!` restarts it, so
reinitialising replays a run exactly.

**Changing the state.** Assigning `integ.u` (then `derivative_discontinuity!(integ, true)`)
or calling `set_u!(integ, u)` changes the population to the new counts before the next
step: in a bin (or stage) that must lose individuals, that many are removed uniformly at
random among those in it; in one that must gain, new individuals are added with traits
uniform in the bin. So a perturbation acts at the resolution of the mesh.
"""
struct IBMProblem{R,D,T<:Real} <: SciMLBase.AbstractSciMLProblem
    rates::R
    u0::Vector{Float64}
    tspan::Tuple{T,T}
    domain::D
    dt::Float64
    seed::UInt64
    eviction::Symbol
end

_stage_model(r) = r isa StageVitalRates
_nstates(r, domain) = _stage_model(r) ? size(r.Qtrans, 1) : domain.n_meshpoints

function IBMProblem(rates, u0, tspan; domain=nothing, dt::Real=1, seed::Integer=0,
                    eviction::Symbol=:reflect)
    rates isa Union{VitalRates,CTVitalRates,StageVitalRates} ||
        throw(ArgumentError("rates must be VitalRates, CTVitalRates or StageVitalRates, got $(typeof(rates))"))
    _stage_model(rates) || domain !== nothing ||
        throw(ArgumentError("a size-structured model needs `domain` (the mesh its state is binned on)"))
    rates isa VitalRates && dt != 1 &&
        throw(ArgumentError("the discrete-time IBM advances one generation per step: dt must be 1, got $dt"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    eviction in (:reflect, :clamp, :discard) ||
        throw(ArgumentError("eviction must be :reflect, :clamp or :discard, got :$eviction"))
    n = _nstates(rates, domain)
    length(u0) == n ||
        throw(ArgumentError("u0 has $(length(u0)) entries, but the model has $n $(_stage_model(rates) ? "stages" : "mesh bins")"))
    all(x -> x >= 0, u0) || throw(ArgumentError("u0 holds counts, which cannot be negative"))
    t0, t1 = tspan isa Real ? (zero(tspan), tspan) : tspan
    T = promote_type(typeof(t0), typeof(t1))
    return IBMProblem(rates, Vector{Float64}(u0), (T(t0), T(t1)), domain, Float64(dt),
                      UInt64(seed), eviction)
end

"""
    remake(prob::IBMProblem; rates, u0, tspan, domain, dt, seed, eviction)

A copy of the problem with some fields changed, for example `remake(prob; seed=2)` for
another replicate or `remake(prob; u0=n)` for another initial population.
"""
function SciMLBase.remake(prob::IBMProblem; rates=prob.rates, u0=prob.u0, tspan=prob.tspan,
                          domain=prob.domain, dt=prob.dt, seed=prob.seed,
                          eviction=prob.eviction)
    return IBMProblem(rates, u0, tspan; domain, dt, seed, eviction)
end

function Base.show(io::IO, p::IBMProblem)
    kind = p.rates isa VitalRates ? "discrete-time" :
           p.rates isa CTVitalRates ? "continuous-time" : "stage-structured"
    return print(io, "IBMProblem (", kind, ", ", length(p.u0), " states, tspan = ", p.tspan,
                 ", seed = ", p.seed, ")")
end

# --- worlds ---------------------------------------------------------------------------------

_bin(z, domain) = clamp(floor(Int, (z - domain.lower) /
                                  ((domain.upper - domain.lower) / domain.n_meshpoints)) + 1,
                        1, domain.n_meshpoints)
_trait_in_bin(rng, b, domain) = domain.lower +
                                (b - 1 + rand(rng)) * (domain.upper - domain.lower) /
                                domain.n_meshpoints

# stochastic rounding keeping the expectation
_round_count(rng, x) = (k = floor(Int, x); k + (rand(rng) < x - k))

function _new_world(prob::IBMProblem, rng)
    r = prob.rates
    world = if r isa VitalRates
        ibm_world(r.survival, r.growth, r.fecundity, prob.domain; rng, eviction=prob.eviction)
    elseif r isa CTVitalRates
        ibm_world_ct(r.flow, r.mortality, r.fecundity, r.recruit, prob.domain; rng,
                     eviction=prob.eviction)
    else
        ibm_world_stage(r.Qtrans, r.death, r.birth; rng)
    end
    _add!(world, prob, rng, [_round_count(rng, x) for x in prob.u0])
    return world
end

# add k[b] individuals to each bin or stage b
function _add!(world, prob::IBMProblem, rng, k)
    for b in eachindex(k), _ in 1:k[b]
        if _stage_model(prob.rates)
            Ark.new_entity!(world, (Stage(b),))
        else
            Ark.new_entity!(world, (Size(_trait_in_bin(rng, b, prob.domain)),))
        end
    end
    return world
end

function _counts(world, prob::IBMProblem)
    _stage_model(prob.rates) && return Float64.(stage_counts(world, length(prob.u0)))
    return Float64.(trait_histogram(world, prob.domain))
end

# the living individuals of each bin or stage
function _members(world, prob::IBMProblem)
    groups = [Ark.Entity[] for _ in eachindex(prob.u0)]
    if _stage_model(prob.rates)
        for q in Ark.Query(world, (Stage,))
            eids, stages = q
            for i in eachindex(eids)
                push!(groups[stages[i].s], eids[i])
            end
        end
    else
        for q in Ark.Query(world, (Size,))
            eids, sizes = q
            for i in eachindex(eids)
                push!(groups[_bin(sizes[i].z, prob.domain)], eids[i])
            end
        end
    end
    return groups
end

# change the population to the counts `target` (rounded stochastically)
function _apply_counts!(world, prob::IBMProblem, rng, target)
    length(target) == length(prob.u0) ||
        throw(ArgumentError("the state has $(length(prob.u0)) entries, got $(length(target))"))
    all(x -> x >= 0, target) || throw(ArgumentError("counts cannot be negative"))
    groups = _members(world, prob)
    gain = zeros(Int, length(groups))
    doomed = Ark.Entity[]
    for b in eachindex(groups)
        d = _round_count(rng, target[b]) - length(groups[b])
        if d > 0
            gain[b] = d
        elseif d < 0
            g = groups[b]
            for j in 1:(-d)                       # partial Fisher-Yates: -d of them at random
                k = j + floor(Int, rand(rng) * (length(g) - j + 1))
                g[j], g[k] = g[k], g[j]
                push!(doomed, g[j])
            end
        end
    end
    for e in doomed
        Ark.is_alive(world, e) && Ark.remove_entity!(world, e)
    end
    return _add!(world, prob, rng, gain)
end

function _advance!(world, prob::IBMProblem)
    r = prob.rates
    r isa VitalRates ? ibm_step!(world) :
    r isa CTVitalRates ? ibm_step_ct!(world, prob.dt) : ibm_step_stage!(world, prob.dt)
    return world
end

# --- integrator -----------------------------------------------------------------------------

"""
    IBMIntegrator

The stepping state of an [`IBMProblem`](@ref), created by `init(prob, IBMStepping(); ...)`,
with the interface of a SciML integrator:

- `integ.u`: the binned (or staged) counts, a `Vector{Float64}`; `integ.t`: the time;
  `integ.world`: the Ark world of individuals.
- `step!(integ)` (one step of `dt`), `step!(integ, n)` (`n` steps),
  `step!(integ, dt, stop_at_tdt)` (until the time has advanced by at least `dt`, or exactly
  `dt`, which must then be a whole number of steps), `solve!(integ)`, `reinit!(integ)`,
  `set_u!(integ, u)`, `terminate!(integ)`, `check_error(integ)`.
- Changing `integ.u` in place takes effect before the next step once
  `derivative_discontinuity!(integ, true)` (or `u_modified!`) is called.
"""
mutable struct IBMIntegrator{P<:IBMProblem,W}
    prob::P
    alg::IBMStepping
    world::W
    rng::Random.Xoshiro
    t::Float64
    retcode::SciMLBase.ReturnCode.T
    saveat::Any
    save_everystep::Bool
    save_end::Bool
    sol_u::Vector{Vector{Float64}}
    sol_t::Vector{Float64}
    _u::Vector{Float64}
    _fresh::Bool
    _modified::Bool
end

function Base.getproperty(integ::IBMIntegrator, name::Symbol)
    name === :u || return getfield(integ, name)
    if !getfield(integ, :_fresh)
        setfield!(integ, :_u, _counts(getfield(integ, :world), getfield(integ, :prob)))
        setfield!(integ, :_fresh, true)
    end
    return getfield(integ, :_u)
end
function Base.setproperty!(integ::IBMIntegrator, name::Symbol, v)
    if name === :u
        length(v) == length(integ.prob.u0) ||
            throw(ArgumentError("the state has $(length(integ.prob.u0)) entries, got $(length(v))"))
        setfield!(integ, :_u, Vector{Float64}(v))
        setfield!(integ, :_fresh, true)
        setfield!(integ, :_modified, true)
        return v
    end
    return setfield!(integ, name, convert(fieldtype(typeof(integ), name), v))
end
Base.propertynames(::IBMIntegrator) = (:u, :t, :prob, :alg, :world, :retcode)

function Base.show(io::IO, integ::IBMIntegrator)
    return print(io, "IBMIntegrator at t = ", integ.t, " (", sum(integ.u), " individuals)")
end

function _sync!(integ::IBMIntegrator)
    if getfield(integ, :_modified)
        _apply_counts!(integ.world, integ.prob, integ.rng, getfield(integ, :_u))
        setfield!(integ, :_fresh, false)
        setfield!(integ, :_modified, false)
    end
    return nothing
end

function _should_save(integ::IBMIntegrator)
    integ.save_end && integ.t >= integ.prob.tspan[2] - 1e-9 * integ.prob.dt && return true
    s = integ.saveat
    s === nothing && return integ.save_everystep
    t0 = integ.prob.tspan[1]
    if s isa Real
        k = round((integ.t - t0) / s)
        return abs(integ.t - t0 - k * s) <= 1e-9 * max(1.0, abs(integ.t))
    end
    return any(x -> abs(x - integ.t) <= 1e-9 * max(1.0, abs(x)), s)
end

function _save!(integ::IBMIntegrator)
    push!(integ.sol_u, copy(integ.u))
    push!(integ.sol_t, integ.t)
    return nothing
end

"""
    init(prob::IBMProblem, alg=IBMStepping(); seed=nothing, saveat=nothing,
         save_everystep=saveat === nothing, save_start=true, save_end=true)

Set up an [`IBMProblem`](@ref) for stepping. `seed` runs with that seed instead of the
problem's; `saveat` is a spacing or a collection of times at which `solve!` records the
state.
"""
function CommonSolve.init(prob::IBMProblem, alg::IBMStepping=IBMStepping(); seed=nothing,
                          saveat=nothing, save_everystep::Bool=saveat === nothing,
                          save_start::Bool=true, save_end::Bool=true)
    seed === nothing || (prob = SciMLBase.remake(prob; seed=seed))
    saveat isa Real && saveat <= 0 && throw(ArgumentError("saveat must be positive, got $saveat"))
    rng = Random.Xoshiro(prob.seed)
    world = _new_world(prob, rng)
    integ = IBMIntegrator(prob, alg, world, rng, Float64(prob.tspan[1]),
                          SciMLBase.ReturnCode.Default, saveat, save_everystep, save_end,
                          Vector{Float64}[], Float64[], Float64[], false, false)
    save_start && _save!(integ)
    return integ
end

_finished(integ::IBMIntegrator) = integ.t >= integ.prob.tspan[2] - 1e-9 * integ.prob.dt ||
                                  integ.retcode == SciMLBase.ReturnCode.Terminated

function CommonSolve.step!(integ::IBMIntegrator, n::Integer=1)
    for _ in 1:n
        _finished(integ) && break
        _sync!(integ)
        _advance!(integ.world, integ.prob)
        k = round(Int, (integ.t - integ.prob.tspan[1]) / integ.prob.dt) + 1
        setfield!(integ, :t, integ.prob.tspan[1] + k * integ.prob.dt)
        setfield!(integ, :_fresh, false)
        _should_save(integ) && _save!(integ)
    end
    return integ
end

function CommonSolve.step!(integ::IBMIntegrator, dt::Real, stop_at_tdt::Bool)
    dt >= 0 || throw(ArgumentError("dt must be non-negative, got $dt"))
    steps = dt / integ.prob.dt
    whole = round(steps)
    if stop_at_tdt && abs(steps - whole) > 1e-9 * max(1.0, steps)
        throw(ArgumentError("the model steps by $(integ.prob.dt), so it cannot stop exactly after dt = $dt"))
    end
    return step!(integ, stop_at_tdt ? Int(whole) : ceil(Int, steps - 1e-9))
end

function CommonSolve.solve!(integ::IBMIntegrator)
    while !_finished(integ)
        step!(integ)
    end
    integ.retcode == SciMLBase.ReturnCode.Default &&
        (integ.retcode = SciMLBase.ReturnCode.Success)
    return IBMSolution(integ.sol_u, integ.sol_t, integ.prob, integ.retcode)
end

"""
    reinit!(integ::IBMIntegrator, u0=integ.prob.u0; t0=integ.prob.tspan[1])

Restart from the counts `u0` at time `t0` with a fresh world and the random number generator
restarted from the problem's seed, discarding the saved states: reinitialising with the same
`u0` and `t0` replays the run exactly.
"""
function SciMLBase.reinit!(integ::IBMIntegrator, u0=integ.prob.u0; t0::Real=integ.prob.tspan[1])
    prob = u0 === integ.prob.u0 ? integ.prob : SciMLBase.remake(integ.prob; u0=u0)
    rng = Random.Xoshiro(prob.seed)
    setfield!(integ, :prob, prob)
    setfield!(integ, :rng, rng)
    setfield!(integ, :world, _new_world(prob, rng))
    setfield!(integ, :t, Float64(t0))
    integ.retcode = SciMLBase.ReturnCode.Default
    setfield!(integ, :_fresh, false)
    setfield!(integ, :_modified, false)
    empty!(integ.sol_u)
    empty!(integ.sol_t)
    _save!(integ)
    return integ
end

"""
    set_u!(integ::IBMIntegrator, u)

Change the population to the counts `u` before the next step, by removing or adding
individuals bin by bin (or stage by stage); see [`IBMProblem`](@ref).
"""
function SciMLBase.set_u!(integ::IBMIntegrator, u)
    integ.u = u
    return nothing
end

"""
    u_modified!(integ::IBMIntegrator, modified::Bool)

Tell the integrator whether `integ.u` was changed in place, so that the change is (or is
not) applied to the individuals before the next step. `derivative_discontinuity!` (SciMLBase
3's name) does the same.
"""
function SciMLBase.u_modified!(integ::IBMIntegrator, modified::Bool)
    modified && getproperty(integ, :u)          # make sure the copy being modified is current
    setfield!(integ, :_modified, modified)
    return nothing
end
function SciMLBase.derivative_discontinuity!(integ::IBMIntegrator, modified::Bool)
    return SciMLBase.u_modified!(integ, modified)
end

"""
    terminate!(integ::IBMIntegrator, retcode=ReturnCode.Terminated)

Stop the run after the current step, for example when the population is extinct.
"""
function SciMLBase.terminate!(integ::IBMIntegrator,
                              retcode::SciMLBase.ReturnCode.T=SciMLBase.ReturnCode.Terminated)
    integ.retcode = retcode
    return nothing
end

"""
    check_error(integ::IBMIntegrator) -> ReturnCode.T

`ReturnCode.Success` while the run is going normally, else its return code (for example
`Terminated`).
"""
function SciMLBase.check_error(integ::IBMIntegrator)
    return integ.retcode == SciMLBase.ReturnCode.Default ? SciMLBase.ReturnCode.Success :
           integ.retcode
end

# --- solutions and ensembles --------------------------------------------------------------

"""
    IBMSolution

The result of `solve(prob::IBMProblem, ...)`: the saved states `u` (binned or staged counts)
at times `t`, the problem, and the `retcode`.
"""
struct IBMSolution{P<:IBMProblem}
    u::Vector{Vector{Float64}}
    t::Vector{Float64}
    prob::P
    retcode::SciMLBase.ReturnCode.T
end

Base.length(sol::IBMSolution) = length(sol.u)
function Base.show(io::IO, sol::IBMSolution)
    return print(io, "IBMSolution (", length(sol.t), " saved states, retcode ", sol.retcode, ")")
end

"""
    successful_retcode(sol::IBMSolution) -> Bool

Whether the run ended normally (`Success` or `Terminated`).
"""
SciMLBase.successful_retcode(sol::IBMSolution) = SciMLBase.successful_retcode(sol.retcode)

"""
    solve(prob::IBMProblem, alg=IBMStepping(); kwargs...) -> IBMSolution

Run the individual-based model over its `tspan` (`init` then `solve!`; keywords as for
`init`).
"""
function CommonSolve.solve(prob::IBMProblem, alg::IBMStepping=IBMStepping(); kwargs...)
    return CommonSolve.solve!(CommonSolve.init(prob, alg; kwargs...))
end

"""
    seed_trajectory(prob::IBMProblem, ctx)

The default `prob_func` of an `EnsembleProblem` of an [`IBMProblem`](@ref): trajectory `i`
is `remake(prob; seed=prob.seed + i)`, or uses the per-trajectory seed SciMLBase derives when
the ensemble is solved with a `seed` keyword.
"""
function seed_trajectory(prob::IBMProblem, ctx)
    seed = ctx.sim_seed === nothing ? prob.seed + UInt64(ctx.sim_id) : UInt64(ctx.sim_seed)
    return SciMLBase.remake(prob; seed=seed)
end

"""
    EnsembleProblem(prob::IBMProblem; prob_func=seed_trajectory, kwargs...)

An ensemble of independent replicates: by default each trajectory gets its own seed
([`seed_trajectory`](@ref)). Solve it with `solve(ens, IBMStepping(), EnsembleThreads();
trajectories)`.
"""
function SciMLBase.EnsembleProblem(prob::IBMProblem; prob_func=seed_trajectory,
                                   safetycopy::Bool=false, kwargs...)
    return invoke(SciMLBase.EnsembleProblem, Tuple{Any}, prob; prob_func, safetycopy, kwargs...)
end
