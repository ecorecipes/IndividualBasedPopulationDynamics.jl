using Test
using IndividualBasedPopulationDynamics
using IntegralProjectionModels          # vital-rate types, samplers, IPMProblem/solve
import StructuredPopulationCore as SPC
using Random
using Statistics

@testset "IndividualBasedPopulationDynamics" begin
    # Vital rates with z-independent survival/fecundity (so the deterministic total
    # is exact regardless of mesh), but z-dependent growth + a recruit distribution.
    s = LinearSurvival(1.0, 0.0)                  # logistic(1.0) ≈ 0.731 survival
    g = NormalGrowth(0.5, 0.8, 0.4)              # grow toward 0.5 + 0.8 z
    f = FecundityRate(-1.0, 0.0, 1.0, 0.3, 1.0)  # ≈ exp(-1) = 0.368 offspring; recruit N(1, 0.3)
    dom = SPC.ContinuousDomain(0.0, 6.0, 60)
    kernel = PKernel(s, g, dom) + FKernel(f, dom)
    z = SPC.meshpoints(dom)
    h = SPC.step_size(dom)
    nbins = length(z)

    rng = Random.Xoshiro(20240611)
    N0 = 20000
    traits0 = clamp.(2.0 .+ 0.5 .* randn(rng, N0), 0.01, 5.99)

    # deterministic IPM seeded from the same initial individuals (binned)
    n0 = zeros(nbins)
    for zt in traits0
        b = clamp(floor(Int, (zt - dom.lower) / h) + 1, 1, nbins)
        n0[b] += 1
    end
    detsol = solve(IPMProblem(kernel, dom, n0, (0, 2)), DirectIteration())
    det_total = [sum(u) for u in detsol.u]

    @testset "large-N realization tracks the deterministic IPM" begin
        w = ibm_world(s, g, f, dom; rng=rng, traits0=collect(traits0), eviction=:reflect)
        res = ibm_run!(w, 2; save_traits=true)
        @test res.N[1] == N0
        @test isapprox(res.N[2], det_total[2]; rtol=0.03)   # totals (LLN)
        @test isapprox(res.N[3], det_total[3]; rtol=0.04)

        n1 = detsol.u[2]                                     # mean trait after one step
        det_mean1 = sum(z .* n1) / sum(n1)
        @test isapprox(mean(res.traits[2]), det_mean1; atol=0.08)

        @test all(x -> 0.0 <= x <= 6.0, res.traits[end])    # eviction keeps traits in-domain
    end

    @testset "small-N stochasticity and extinction" begin
        s2 = LinearSurvival(-1.0, 0.0)                       # logistic(-1) ≈ 0.27
        f2 = FecundityRate(-50.0, 0.0, 1.0, 0.3, 1.0)        # ~0 offspring -> subcritical
        w = ibm_world(s2, g, f2, dom; rng=Random.Xoshiro(7), traits0=fill(2.0, 30))
        res = ibm_run!(w, 40)
        @test res.N[end] == 0
        @test population_size(w) == 0
    end

    @testset "independent realizations differ" begin
        w1 = ibm_world(s, g, f, dom; rng=Random.Xoshiro(1), traits0=fill(2.0, 200))
        w2 = ibm_world(s, g, f, dom; rng=Random.Xoshiro(2), traits0=fill(2.0, 200))
        ibm_run!(w1, 3)
        ibm_run!(w2, 3)
        @test population_size(w1) != population_size(w2) ||
              sort(traits(w1)) != sort(traits(w2))
        @test sum(trait_histogram(w1, dom)) == population_size(w1)
    end
end
