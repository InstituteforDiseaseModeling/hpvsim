"""
Plot implied natural history.
"""
import hpvsim as hpv
import hpvsim.parameters as hppar
import hpvsim.utils as hpu
import pylab as pl
import pandas as pd
from scipy.stats import lognorm, norm
import numpy as np
import sciris as sc
import seaborn as sns
import math


# %% Functions


class cum_dist(hpv.Analyzer):
    '''
    Determine distribution of time to clearance, persistence, pre-cancer and cancer
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year

    def initialize(self, sim):
        super().initialize(sim)
        if self.start_year is None:
            self.start_year = sim['start']
        self.dur_to_clearance = []
        self.dur_to_cin = []
        self.dur_to_cancer = []
        self.total_infections = 0


    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            inf_genotypes, inf_inds = (sim.people.date_exposed == sim.t).nonzero()
            self.total_infections += len(inf_inds)
            if len(inf_inds):
                infs_that_progress_bools = hpv.utils.defined(sim.people.date_cin[inf_genotypes, inf_inds])
                infs_that_progress_inds = hpv.utils.idefined(sim.people.date_cin[inf_genotypes, inf_inds], inf_inds)
                infs_to_cancer_bools = hpv.utils.defined(sim.people.date_cancerous[inf_genotypes, inf_inds])
                infs_to_cancer_inds = hpv.utils.idefined(sim.people.date_cancerous[inf_genotypes, inf_inds], inf_inds)
                infs_that_clear_bools = hpv.utils.defined(sim.people.date_clearance[inf_genotypes, inf_inds])
                infs_that_clear_inds = hpv.utils.idefined(sim.people.date_clearance[inf_genotypes, inf_inds], inf_inds)

                dur_to_clearance = (sim.people.date_clearance[inf_genotypes[infs_that_clear_bools], infs_that_clear_inds] - sim.t)*sim['dt']
                dur_to_cin = (sim.people.date_cin[inf_genotypes[infs_that_progress_bools], infs_that_progress_inds] - sim.t)*sim['dt']
                dur_to_cancer = (sim.people.date_cancerous[inf_genotypes[infs_to_cancer_bools], infs_to_cancer_inds] - sim.t)*sim['dt']

                self.dur_to_clearance += dur_to_clearance.tolist()
                self.dur_to_cin += dur_to_cin.tolist()
                self.dur_to_cancer += dur_to_cancer.tolist()


class outcomes_by_year(hpv.Analyzer):
    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.durations = np.arange(0,31)
        result_keys = ['cleared', 'persisted', 'progressed', 'cancer', 'dead', 'total']
        self.results = {rkey: np.zeros_like(self.durations) for rkey in result_keys}

    def initialize(self, sim):
        super().initialize(sim)
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] == self.start_year:
            idx = ((sim.people.date_exposed == sim.t) & (sim.people.sex==0)).nonzero()  # Get people exposed on this step
            inf_inds = idx[-1]
            if len(inf_inds):
                time_to_clear = (sim.people.date_clearance[idx] - sim.t)*sim['dt']
                time_to_cancer = (sim.people.date_cancerous[idx] - sim.t)*sim['dt']
                time_to_cin = (sim.people.date_cin[idx] - sim.t)*sim['dt']

                # Count deaths. Note that there might be more people with a defined
                # cancer death date than with a defined cancer date because this is
                # counting all death, not just deaths resulting from infections on this
                # time step.
                time_to_cancer_death = (sim.people.date_dead_cancer[inf_inds] - sim.t)*sim['dt']
                time_to_other_death = (sim.people.date_dead_other[inf_inds] - sim.t)*sim['dt']

                for dd in self.durations:

                    dead = (time_to_cancer_death <= (dd+1)) | (time_to_other_death <= (dd+1))
                    cleared = ~dead & (time_to_clear <= (dd+1))
                    persisted = ~dead & ((time_to_clear > (dd+1)) | (time_to_cancer > (dd+1)))
                    persisted_no_progression = persisted & ~(time_to_cin <= (dd+1))  # USing the ~ means that we also count nans
                    persisted_with_progression = persisted & (time_to_cin <= (dd+1))
                    cancer = ~dead & (time_to_cancer <= (dd+1))

                    derived_total = len(hpv.true(cleared)) + len(hpv.true(persisted_no_progression)) + len(hpv.true(persisted_with_progression)) + len(hpv.true(cancer)) + len(hpv.true(dead))

                    if derived_total != len(inf_inds):
                        errormsg = "Something is wrong!"
                        raise ValueError(errormsg)

                    self.results['cleared'][dd] += len(hpv.true(cleared))
                    self.results['persisted'][dd] += len(hpv.true(persisted_no_progression))
                    self.results['progressed'][dd] += len(hpv.true(persisted_with_progression))
                    self.results['cancer'][dd] += len(hpv.true(cancer))
                    self.results['dead'][dd] += len(hpv.true(dead))
                    self.results['total'][dd] += derived_total


class dwelltime_by_genotype(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = dict()
        self.age_cancer = dict()
        self.dwelltime = dict()
        self.median_age_causal = dict()
        for gtype in range(sim['n_genotypes']):
            self.age_causal[gtype] = []
            self.age_cancer[gtype] = []
        for state in ['precin', 'cin', 'total']:
            self.dwelltime[state] = dict()
            for gtype in range(sim['n_genotypes']):
                self.dwelltime[state][gtype] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                dur_precin = sim.people.dur_precin[cancer_genotypes, cancer_inds]
                dur_cin = sim.people.dur_cin[cancer_genotypes, cancer_inds]
                total_time = (sim.t - date_exposed) * sim['dt']
                for gtype in range(sim['n_genotypes']):
                    gtype_inds = hpv.true(cancer_genotypes == gtype)
                    self.dwelltime['precin'][gtype] += dur_precin[gtype_inds].tolist()
                    self.dwelltime['cin'][gtype] += dur_cin[gtype_inds].tolist()
                    self.dwelltime['total'][gtype] += total_time[gtype_inds].tolist()
                    self.age_causal[gtype] += (current_age[gtype_inds] - total_time[gtype_inds]).tolist()
                    self.age_cancer[gtype] += (current_age[gtype_inds]).tolist()
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''
        for gtype in range(sim['n_genotypes']):
            self.median_age_causal[gtype] = np.quantile(self.age_causal[gtype], 0.5)


def set_font(size=None):
    ''' Set a custom font '''
    sc.options(fontsize=size)
    return

def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale

def plot_nh(sim=None):

    res = sim.analyzers[1].results
    years = sim.analyzers[1].durations

    df = pd.DataFrame()
    df['years'] = years
    df['prob_clearance'] = res['cleared']/res['total'] * 100
    df['prob_persist'] = res['persisted']/res['total'] * 100
    df['prob_progressed'] = res['progressed']/res['total'] * 100
    df['prob_cancer'] = res['cancer']/res['total'] * 100
    df['prob_dead'] = res['dead']/res['total'] * 100

    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=16)
    colors = sc.gridcolors(5)
    fig, ax = pl.subplots(figsize=(11, 9))
    bottom = np.zeros(len(df['years']))
    layers = ['prob_clearance', 'prob_persist', 'prob_progressed', 'prob_cancer', 'prob_dead']
    labels = ['Cleared', 'Persisted', 'Progressed', 'Cancer', 'Dead']
    for ln,layer in enumerate(layers):
        ax.fill_between(df['years'], bottom, bottom+df[layer], color=colors[ln], label=labels[ln])
        bottom += df[layer]
    data_years = np.arange(0,6, 0.5)
    cleared = [0, 58, 68, 71, 78, 81, 83, 84, 84.5, 85, 85.6, 86]
    progressed = [100, 99, 97, 96, 96, 95, 94, 93, 92, 91.5, 91.5, 91]
    ax.scatter(data_years, cleared, color=colors[0])
    ax.scatter(data_years, progressed, color=colors[2])
    ax.legend(loc='lower right')
    ax.set_xlabel('Time since infection')
    fig.tight_layout()
    fig.savefig(f'dist_infections.png')
    fig.show()

    # Make sims
    genotypes = ['hpv16', 'hpv18', 'hi5']
    glabels = ['HPV16', 'HPV18', 'HI5']

    dur_cin = sc.autolist()
    cancer_fns = sc.autolist()
    cin_fns = sc.autolist()
    dur_precin = sc.autolist()
    for gi, genotype in enumerate(genotypes):
        dur_precin += sim['genotype_pars'][genotype]['dur_precin']
        dur_cin += sim['genotype_pars'][genotype]['dur_cin']
        cancer_fns += sim['genotype_pars'][genotype]['cancer_fn']
        cin_fns += sim['genotype_pars'][genotype]['cin_fn']


    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    colors = sc.gridcolors(len(genotypes))
    fig, axes = pl.subplots(2, 3, figsize=(11, 9))
    axes = axes.flatten()
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])

    ####################
    # Make plots
    ####################
    dt = 0.25
    this_precinx = np.arange(dt, 15+dt, dt)
    this_cinx = np.arange(dt, 30+dt, dt)
    n_samples = 10
    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):

        # Panel A: durations of infection
        sigma, scale = lognorm_params(dur_precin[gi]['par1'], dur_precin[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        axes[0].plot(this_precinx, rv.pdf(this_precinx), color=colors[gi], lw=2, label=glabels[gi])

        # Panel B: prob of dysplasia
        dysp = hppar.compute_severity(this_precinx[:], pars=cin_fns[gi])
        axes[1].plot(this_precinx, dysp, color=colors[gi], lw=2, label=gtype.upper())

        # Panel C: durations of CIN
        sigma, scale = lognorm_params(dur_cin[gi]['par1'], dur_cin[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        axes[2].plot(this_cinx, rv.pdf(this_cinx), color=colors[gi], lw=2, label=glabels[gi])

        # Panel D: dysplasia
        cancer = hppar.compute_severity(this_cinx[:], pars=cancer_fns[gi])
        axes[3].plot(this_cinx, cancer, color=colors[gi], lw=2, label=gtype.upper())


    axes[0].set_ylabel("")
    axes[0].grid()
    axes[0].set_xlabel("Duration of infection (years)")
    axes[0].set_title("Distribution of\n infection duration")
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Probability of CIN")
    axes[1].set_xlabel("Duration of infection (years)")
    axes[1].set_title("Infection duration to progression\nfunction")
    axes[1].set_ylim([0,1])
    axes[1].grid()

    axes[2].set_ylabel("")
    axes[2].grid()
    axes[2].set_xlabel("Duration of CIN (years)")
    axes[2].set_title("Distribution of\n CIN duration")
    axes[2].legend(frameon=False)

    axes[3].set_ylim([0,1])
    axes[3].grid()
    # axes[2].set_ylabel("Probability of transformation")
    axes[3].set_xlabel("Duration of CIN (years)")
    axes[3].set_title("Probability of cancer\n within X years")

    # Panel F: CIN dwelltime
    a = sim.get_analyzer('dwelltime_by_genotype')
    dd = {}
    dd['dwelltime'] = sc.autolist()
    dd['genotype'] = sc.autolist()
    dd['state'] = sc.autolist()
    for cin in ['precin', 'cin']:
        dt = a.dwelltime[cin]
        data = dt[0]+dt[1]+dt[2]
        labels = ['HPV16']*len(dt[0]) + ['HPV18']*len(dt[1]) + ['HI5']*len(dt[2])
        dd['dwelltime'] += data
        dd['genotype'] += labels
        dd['state'] += [cin.upper()]*len(labels)
    df = pd.DataFrame(dd)
    sns.boxplot(data=df, x="state", y="dwelltime", hue="genotype", ax=axes[5], showfliers=False, palette=colors)
    axes[5].legend([], [], frameon=False)
    axes[5].set_xlabel("")
    axes[5].set_ylabel("Dwelltime")
    axes[5].set_title('Dwelltimes from\n infection to CIN grades')

    ac = {}
    ac['age'] = sc.autolist()
    ac['genotype'] = sc.autolist()
    ac['state'] = sc.autolist()
    for state, state_label in zip([a.age_causal, a.age_cancer], ['infection', 'cancer']):
        data = state[0]+state[1]+state[2]
        labels = ['HPV16']*len(state[0]) + ['HPV18']*len(state[1]) + ['HI5']*len(state[2])
        ac['age'] += data
        ac['genotype'] += labels
        ac['state'] += [state_label.upper()]*len(labels)
    ac_df = pd.DataFrame(ac)
    sns.boxplot(data=ac_df, x="state", y="age", hue="genotype", ax=axes[4], showfliers=False, palette=colors)
    axes[4].legend([], [], frameon=False)
    axes[4].set_xlabel("")
    axes[4].set_ylabel("Age")
    axes[4].set_title('Age of causal infection\nand cancer')

    fig.tight_layout()
    fig.savefig(f'nathx.png')
    fig.show()

    return


def plot_nh_simple(sim=None):
    # Make sims
    genotypes = ['hpv16', 'hpv18', 'hi5']
    glabels = ['HPV16', 'HPV18', 'HI5']

    dur_cin = sc.autolist()
    cancer_fns = sc.autolist()
    cin_fns = sc.autolist()
    dur_precin = sc.autolist()
    for gi, genotype in enumerate(genotypes):
        dur_precin += sim['genotype_pars'][genotype]['dur_precin']
        dur_cin += sim['genotype_pars'][genotype]['dur_cin']
        cancer_fns += sim['genotype_pars'][genotype]['cancer_fn']
        cin_fns += sim['genotype_pars'][genotype]['cin_fn']


    ####################
    # Make figure, set fonts and colors
    ####################
    set_font(size=12)
    colors = sc.gridcolors(len(genotypes))
    fig, axes = pl.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])

    ####################
    # Make plots
    ####################
    dt = 0.25
    this_precinx = np.arange(dt, 15+dt, dt)
    years = np.arange(1,16,1)
    this_cinx = np.arange(dt, 30+dt, dt)
    n_samples = 10

    width = .3
    multiplier=0

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        offset = width * multiplier

        # Panel A: durations of infection
        # axes[0].set_ylim([0,1])
        if gi == 0:
            s_16, scale_16 = hpu.logn_percentiles_to_pars(1, 0.5, 3, 0.6)
            rv = lognorm(s=s_16, scale=scale_16)
        else:
            s, scale = hpu.logn_percentiles_to_pars(1, 0.7, 3, 0.86)
            rv = lognorm(s=s, scale=scale)
        axes[0].bar(years+offset - width/3, 1-rv.cdf(years), color=colors[gi], lw=2, label=glabels[gi], width=width)
        multiplier += 1
        # Panel B: prob of dysplasia
        dysp = hppar.compute_severity(this_precinx[:], pars=cin_fns[gi])
        axes[1].plot(this_precinx, dysp, color=colors[gi], lw=2, label=gtype.upper())

        # Panel C: durations of CIN
        sigma, scale = lognorm_params(dur_cin[gi]['par1'], dur_cin[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        axes[2].plot(this_cinx, rv.pdf(this_cinx), color=colors[gi], lw=2, label=glabels[gi])

        # Panel D: dysplasia
        cancer = hppar.compute_severity(this_cinx[:], pars=cancer_fns[gi])
        axes[3].plot(this_cinx, cancer, color=colors[gi], lw=2, label=gtype.upper())


    axes[0].set_ylabel("")
    axes[0].grid()
    axes[0].set_xlabel("Duration of infection (years)")
    axes[0].set_title("Probability of persistance")
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Probability of CIN")
    axes[1].set_xlabel("Duration of infection (years)")
    axes[1].set_title("Infection duration to progression\nfunction")
    axes[1].set_ylim([0,1])
    axes[1].grid()

    axes[2].set_ylabel("")
    axes[2].grid()
    axes[2].set_xlabel("Duration of CIN (years)")
    axes[2].set_title("Distribution of\n CIN duration")
    axes[2].legend(frameon=False)

    axes[3].set_ylim([0,1])
    axes[3].grid()
    # axes[2].set_ylabel("Probability of transformation")
    axes[3].set_xlabel("Duration of CIN (years)")
    axes[3].set_title("Probability of cancer\n within X years")

    fig.tight_layout()
    fig.savefig(f'nathx_simple.png')
    fig.show()
    return

# %% Run as a script
if __name__ == '__main__':

    make_simple = False
    make_stacked = True
    do_run = True

    if make_simple:
        sim = hpv.Sim()
        sim.initialize()
        plot_nh_simple(sim)

    # sim.pars['genotype_pars']['hpv16']['cin_fn']['x_infl']=0
    # sim.pars['genotype_pars']['hpv16']['cin_fn']['k'] = 0.2
    #
    # sim.pars['genotype_pars']['hpv18']['cin_fn']['x_infl']=0
    # sim.pars['genotype_pars']['hpv18']['cin_fn']['k'] = 0.2
    # sim.pars['genotype_pars']['hpv18']['cin_fn']['y_max'] = 0.9
    #
    # sim.pars['genotype_pars']['hi5']['cin_fn']['x_infl']=0
    # sim.pars['genotype_pars']['hi5']['cin_fn']['k'] = 0.2
    # sim.pars['genotype_pars']['hi5']['cin_fn']['y_max'] = 0.85

    if make_stacked:

        location = 'nigeria'
        if do_run:
            pars = {
                'location': location,
                'start': 1970,
                'end': 2020,
                'ms_agent_ratio': 100,
                'n_agents': 50e3,
                # 'sev_dist': dict(dist='normal_pos', par1=1.25, par2=0.2)
            }
            age_causal_by_genotype = dwelltime_by_genotype(start_year=2000)
            inf_dist = outcomes_by_year(start_year=2000)
            sim = hpv.Sim(pars, analyzers=[age_causal_by_genotype, inf_dist])
            sim.run()
            sim.plot()
            sim.shrink()
            sc.saveobj(f'{location}.sim', sim)

        else:
            sim = sc.loadobj(f'{location}.sim')


        plot_nh(sim)

    print('Done.')