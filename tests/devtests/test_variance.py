'''
Check impact on sim variance of population size and variance switch
'''

import numpy as np
import sciris as sc
import hpvsim as hpv

base_pars = dict(
    genotypes = [16],
    verbose = -1,
)

p = sc.objdict(
    minvars = [0, 2],
    popsizes = [10e3, 40e3],
    repeats = 20,
    trials = 2,
    start = 1950,
    end = 2050,
)

offset = 0
randval = np.random.randint(1e5)


if __name__ == '__main__':
    
    T = sc.timer()
    
    ppl = dict()
    for popsize in p.popsizes:
        sim = hpv.Sim(**base_pars, n_agents=popsize)
        sim.initialize()
        ppl[popsize] = sim.people
    
    allsims = []
    for minvar in p.minvars:
        sc.heading(f'Running minvar={minvar}')
        hpv.options(min_var=minvar)
        sims = []
        for trial in range(p.trials):
            for popsize in p.popsizes:
                label = f'minvar{minvar}_popsize{popsize}_trial{trial}'
                for r in range(p.repeats):
                    pars = dict(n_agents=popsize, rand_seed=r+trial*p.repeats+offset*randval)
                    sim = hpv.Sim(**base_pars, **pars, label=f'{label}_r{r}')
                    sim.people = ppl[popsize]
                    sim.info = sc.objdict(minvar=minvar, trial=trial, **pars)
                    sims.append(sim)
        msim = hpv.MultiSim(sims)
        T.tic()
        msim.run(keep_people=True)
        allsims += msim.sims
        T.toc(f'minvar={minvar}')
    
    d = []
    for sim in allsims:
        s = sim.summary
        row = sc.mergedicts(sim.info, cancer=s.total_cancers, inci=s.total_infections)
        d.append(row)
        
    df = sc.dataframe(d)
    
    g = df.groupby(by=['n_agents', 'minvar', 'trial'])
    mean = g.mean()
    std = g.std()
    res = std/mean
    
    sc.heading('Inputs')
    print(p)
    
    sc.heading('Mean')
    print(mean)
    sc.heading('STD')
    print(std)
    sc.heading('Coefficient of variation')
    print(res)
    
    T.toc()