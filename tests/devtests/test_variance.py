'''
Check impact on sim variance of population size and variance switch
'''

import sciris as sc
import pylab as pl
import hpvsim as hpv

repeats = 10
popsizes = [10e3, 100e3]
minvars = [0]

T = sc.timer()

d = sc.objdict()
for minvar in minvars:
    for popsize in popsizes:
        label = f'minvar{minvar}_popsize{popsize}'
        sims = []
        for r in range(repeats):
            pars = dict(n_agents=popsize, rand_seed=r)
            sim = hpv.Sim(**pars, label=f'{label}_r{r}')
            sim.info = sc.objdict(minvar=minvar, **pars)
            sims.append(sim)
        msim = hpv.MultiSim(sims, label=label)
        d[label] = msim

mmsim = hpv.MultiSim.merge(*d.values())
mmsim.run()
msims = mmsim.split()
for k,m in zip(d.keys(), msims):
    d[k] = m
        
T.toc()