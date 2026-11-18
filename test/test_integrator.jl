# The SciML problem/integrator interface (IBMProblem) and its use by the simulation-based
# resilience metrics of DynamicalResilience.jl. Included from runtests.jl.
IBPD = IndividualBasedPopulationDynamics

@testset "IBMProblem and the SciML integrator interface" begin
    s = LinearSurvival(1.0, 0.0)                  # the vital rates of runtests.jl
    g = NormalGrowth(0.5, 0.8, 0.4)
    f = FecundityRate(-1.0, 0.0, 1.0, 0.3, 1.0)
    dom = SPC.ContinuousDomain(0.0, 6.0, 60)
    kernel = PKernel(s, g, dom) + FKernel(f, dom)
    nbins = dom.n_meshpoints
    n0 = zeros(nbins)
    n0[15:25] .= 40.0                                  # 440 individuals, z ≈ 1.4–2.5
    rates = VitalRates(s, g, f)
    prob = IBMProblem(rates, n0, (0, 10); domain=dom, seed=1)

    @testset "construction" begin
        @test_throws ArgumentError IBMProblem(rates, n0, (0, 10))               # no domain
        @test_throws ArgumentError IBMProblem(rates, n0, (0, 10); domain=dom, dt=0.5)
        @test_throws ArgumentError IBMProblem(rates, n0[1:5], (0, 10); domain=dom)
        @test_throws ArgumentError IBMProblem(rates, -n0, (0, 10); domain=dom)
        @test_throws ArgumentError IBMProblem(rates, n0, (0, 10); domain=dom, eviction=:bounce)
        @test_throws ArgumentError IBMProblem((1, 2), n0, (0, 10); domain=dom)
        @test remake(prob; seed=5).seed == 5 && remake(prob; tspan=3).tspan == (0, 3)
        @test occursin("discrete-time", sprint(show, prob))
        @test prob isa SciMLBase.AbstractSciMLProblem
    end

    @testset "init, u and step!" begin
        integ = init(prob)
        @test integ.u == n0 && integ.t == 0            # whole counts are kept exactly
        @test trait_histogram(integ.world, dom) == Int.(n0)
        step!(integ)
        @test integ.t == 1 && sum(integ.u) == population_size(integ.world)
        step!(integ, 3, true)
        @test integ.t == 4
        @test_throws ArgumentError step!(integ, 0.5, true)
        step!(integ, 0.5, false)                       # at least 0.5: one generation
        @test integ.t == 5
        @test SciMLBase.check_error(integ) == SciMLBase.ReturnCode.Success
        step!(integ, 100, true)                        # stops at the end of tspan
        @test integ.t == 10
        terminate!(integ)
        @test SciMLBase.check_error(integ) == SciMLBase.ReturnCode.Terminated
        sol = solve(prob; saveat=5)
        @test sol.t == [0, 5, 10] && SciMLBase.successful_retcode(sol)
        @test sol.u[end] == solve(prob).u[end]         # saving does not touch the randomness
        @test occursin("individuals", sprint(show, init(prob)))
    end

    @testset "exact replay, set_u! and in-place changes" begin
        integ = init(prob)
        step!(integ, 4, true)
        a = copy(integ.u)
        reinit!(integ)
        @test integ.t == 0 && integ.u == n0
        step!(integ, 4, true)
        @test integ.u == a
        # removing individuals bin by bin
        target = floor.(a ./ 2)
        set_u!(integ, target)
        IBPD._sync!(integ)
        @test trait_histogram(integ.world, dom) == Int.(target)
        # adding individuals, with traits in their bins
        up = copy(target)
        up[30] += 7
        set_u!(integ, up)
        IBPD._sync!(integ)
        @test trait_histogram(integ.world, dom) == Int.(up)
        # an in-place change takes effect after derivative_discontinuity!
        integ.u[30] = 0
        SciMLBase.derivative_discontinuity!(integ, true)
        IBPD._sync!(integ)
        @test trait_histogram(integ.world, dom)[30] == 0
        @test_throws ArgumentError set_u!(integ, ones(3))
        # the change is applied before the next step, and the run continues
        set_u!(integ, n0)
        step!(integ)
        @test integ.t == 5 && sum(integ.u) > 0
        # stochastic rounding keeps the expected count
        r = [IBPD._round_count(Random.Xoshiro(k), 2.3) for k in 1:4000]
        @test issubset(unique(r), [2, 3]) && abs(mean(r) - 2.3) < 4 * sqrt(0.21 / 4000)
        # a seed given to init is remake(prob; seed)
        b1 = init(prob; seed=7)
        b2 = init(remake(prob; seed=7))
        step!(b1, 3, true)
        step!(b2, 3, true)
        @test b1.u == b2.u && b1.u != (step!(init(prob), 3, true)).u
    end

    @testset "continuous-time and stage-structured models" begin
        cdom = SPC.ContinuousDomain(0.0, 10.0, 20)
        cp = IBMProblem(CTVitalRates(x -> 0.1, x -> 0.2, x -> 0.25, rng -> 1.0),
                        fill(10.0, 20), (0.0, 2.0); domain=cdom, dt=0.1, seed=2)
        ci = init(cp)
        step!(ci, 0.5, true)
        @test ci.t ≈ 0.5
        @test_throws ArgumentError step!(ci, 0.05, true)
        @test occursin("continuous-time", sprint(show, cp))

        Qtrans = [0.0 0.0 0.0; 0.5 0.0 0.0; 0.0 0.4 0.0]
        death = [0.1, 0.1, 0.2]
        birth = zeros(3, 3)
        birth[1, 3] = 0.6
        sp = IBMProblem(StageVitalRates(Qtrans, death, birth), [100.0, 50.0, 20.0],
                        (0.0, 1.0); dt=0.01, seed=3)
        si = init(sp)
        @test si.u == [100, 50, 20]
        step!(si, 0.2, true)
        @test si.t ≈ 0.2 && si.u == stage_counts(si.world, 3)
        set_u!(si, [0.0, 50.0, 20.0])
        IBPD._sync!(si)
        @test stage_counts(si.world, 3) == [0, 50, 20]
        @test_throws ArgumentError IBMProblem(StageVitalRates(Qtrans, death, birth), [1.0, 2.0],
                                              (0.0, 1.0))
    end

    @testset "ensembles" begin
        sim = solve(SciMLBase.EnsembleProblem(prob), IBMStepping(), SciMLBase.EnsembleSerial();
                    trajectories=3, saveat=10)
        @test [sol.prob.seed for sol in sim.u] == UInt64.(2:4)
        @test length(unique(sol.u[end] for sol in sim.u)) > 1
        seeded = solve(SciMLBase.EnsembleProblem(prob), IBMStepping(),
                       SciMLBase.EnsembleThreads(); trajectories=3, seed=42, saveat=10)
        again = solve(SciMLBase.EnsembleProblem(prob), IBMStepping(),
                      SciMLBase.EnsembleSerial(); trajectories=3, seed=42, saveat=10)
        @test [sol.u[end] for sol in seeded.u] == [sol.u[end] for sol in again.u]
    end

    @testset "DynamicalResilience drives it" begin
        # Paired runs: a pulse that removes every juvenile (z < 1.5). The IBM is density-
        # independent, so the size *distribution* (proportions) converges back to the stable
        # one, at about the rate ln(λ₁/|λ₂|) of the IPM it approximates, until demographic
        # noise (common random numbers end at the pulse) sets a floor of order 1/√N. The fit
        # window stays above that floor.
        big = IBMProblem(rates, n0 .* 20, (0, 22); domain=dom, seed=11)   # 8800 individuals
        proportions(u) = u ./ max(sum(u), 1)
        cull(u, rng) = (v = copy(u); v[1:15] .= 0; v)
        rec = DynamicalResilience.recovery(big, IBMStepping(); perturb=cull,
                                           observe=proportions, t_pulse=8, horizon=14,
                                           interval=1, window=(1.0, 0.08))
        @test rec.retcode == :Success
        @test rec.distance[1] > 0.1                    # a large pulse to the distribution
        @test median(rec.distance[(end - 4):end]) < rec.distance[1] / 4
        ipm = ResilienceCore.linearise(IPMProblem(kernel, dom, n0, (0, 1)))
        ρ = ResilienceCore.resilience(ipm)
        @test isfinite(ρ) && ρ > 0
        @test ρ / 2 < rec.return_rate < 2ρ
        # the replay before the pulse is exact, so the control and perturbed runs differ only
        # through the pulse: the first recorded distance is the pulse itself
        integ = init(big)
        step!(integ, 8, true)
        p = proportions(integ.u)
        @test rec.distance[1] ≈ norm(proportions(cull(integ.u, nothing)) .- p)
    end
end
