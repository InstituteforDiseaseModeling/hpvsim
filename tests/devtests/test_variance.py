'''
Check impact on sim variance of population size and variance switch
'''

import sciris as sc
import hpvsim as hpv

base_pars = dict(
    genotypes = [16],
    verbose = -1,
)

offset = 0
p = sc.objdict(
    minvars = [0, 1],
    popsizes = [10e3, 40e3],
    repeats = 10,
)

T = sc.timer()

allsims = []
for minvar in p.minvars:
    sc.heading(f'Running minvar={minvar}')
    hpv.options(min_var=minvar)
    sims = []
    for popsize in p.popsizes:
        label = f'minvar{minvar}_popsize{popsize}'
        for r in range(p.repeats):
            pars = dict(n_agents=popsize, rand_seed=r+offset)
            sim = hpv.Sim(**base_pars, **pars, label=f'{label}_r{r}')
            sim.info = sc.objdict(minvar=minvar, **pars)
            sims.append(sim)
    msim = hpv.MultiSim(sims)
    msim.run()
    allsims += msim.sims

d = []
for sim in allsims:
    s = sim.summary
    row = sc.mergedicts(sim.info, cancer=s.total_cancers, inci=s.total_infections)
    d.append(row)
    
df = sc.dataframe(d)

g = df.groupby(by=['minvar','n_agents'])
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