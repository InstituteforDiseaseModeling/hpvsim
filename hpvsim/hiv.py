'''
Defines classes and methods for HIV natural history
'''

import numpy as np
import sciris as sc
import pandas as pd
from scipy.stats import weibull_min
from . import utils as hpu
from . import defaults as hpd
from . import base as hpb


class HIVsim(hpb.ParsObj):
    '''
        A class based around performing operations on a self.pars dict.
        '''

    def __init__(self, sim, art_datafile, hiv_datafile, hiv_pars):
        '''
        Initialization
        '''

        # Define some basic settings and attributes
        self.cd4states = ['lt200', 'gt200'] # code names for HIV states
        self.cd4statesfull = ['CD4<200', '200<CD4<500'] # full names for HIV states
        self.cd4_lb = [0, 200] # Lower bound for CD4 states
        self.cd4_ub = [200, 500] # Lower bound for CD4 states
        self.ncd4 = len(self.cd4states)

        # Load in the parameters from provided datafiles
        pars = self.load_data(hiv_datafile=hiv_datafile, art_datafile=art_datafile)

        # Define default parameters, can be overwritten by hiv_pars
        pars['hiv_pars'] = {
            'rel_sus': { # Increased risk of acquiring HPV
                'lt200': 2.2,
                'gt200': 2.2,
                },
            'rel_sev': { # Increased risk of disease severity
                'lt200': 1.5,
                'gt200': 1.2,
                },
            'rel_imm': { # Reduction in neutralizing/t-cell immunity acquired after infection/vaccination
                'lt200': 0.36,
                'gt200': 0.76,
            },
            'rel_reactivation_prob': 3, # Unused for now, TODO: add in rel_reactivation to make functional
            'model_hiv_death': True, # whether or not to model HIV mortality. Typically only set to False for testing purposes
            'time_to_hiv_death_shape': 2, # shape parameter for weibull distribution, based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
            'time_to_hiv_death_scale': lambda a: 21.182 - 0.2717*a, # scale parameter for weibull distribution, based on https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frsif.2013.0613&file=rsif20130613supp1.pdf
            'cd4_start': dict(dist='normal', par1=594, par2=20),
            'cd4_trajectory': lambda f: (24.363 - 16.672*f)**2, # based on https://docs.idmod.org/projects/emod-hiv/en/latest/hiv-model-healthcare-systems.html?highlight=art#art-s-impact-on-cd4-count
            'cd4_reconstitution': lambda m: 15.584*m - 0.2113*m**2, # growth in CD4 count following ART initiation
            'art_failure_prob': 0.1, # Percentage of people on ART who will not suppress virus successfully
            'dt_art': 5.0 # Timestep at which people originally not on ART can initiate care
        }

        self.update_pars(old_pars=pars, new_pars=hiv_pars, create=True)
        self.init_results(sim)

        y = np.linspace(0,1,101)
        cd4_decline = self['hiv_pars']['cd4_trajectory'](y)
        self.cd4_decline_diff = np.diff(cd4_decline)
        sim.pars['hiv_pars'] = sc.mergedicts(sim.pars['hiv_pars'], self.pars)
        return


    def update_pars(self, old_pars=None, new_pars=None, create=True):
        if len(new_pars):
            for parkey, parval in new_pars.items():
                if isinstance(parval, dict):
                    for parvalkey, parvalval in parval.items():
                        if isinstance(parvalval, dict):
                            for parvalkeyval, parvalvalval in parvalval.items():
                                old_pars['hiv_pars'][parkey][parvalkey][parvalkeyval] = parvalvalval
                        else:
                            old_pars['hiv_pars'][parkey][parvalkey] = parvalval
                else:
                    old_pars['hiv_pars'][parkey] = parval

        # Call update_pars() for ParsObj
        super().update_pars(pars=old_pars, create=create)

    @staticmethod
    def init_states(people):
        ''' Add HIV-related states to the people states '''
        hiv_states = [
            hpd.State('hiv', bool, False),
            hpd.State('art', bool, False),
            hpd.State('art_failure', bool, False),
        ]
        people.meta.other_stock_states  += hiv_states
        people.meta.durs                += [hpd.State('dur_hiv',    hpd.default_float,  np.nan)]
        people.meta.person              += [hpd.State('cd4',        hpd.default_float,  np.nan)]
        people.meta.alive_states        += hpd.State('dead_hiv',    bool,               False),

        return


    def init_results(self, sim):

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = hpb.Result(*args, **kwargs, npts=sim.res_npts)
            return output

        self.resfreq = sim.resfreq
        # Initialize storage
        results = sc.objdict()

        na = len(sim['age_bin_edges']) - 1  # Number of age bins

        stock_colors = [i for i in set(sim.people.meta.stock_colors) if i is not None]

        results['hiv_infections'] = init_res('New HIV infections')
        results['hiv_infections_by_age'] = init_res('New HIV infections by age', n_rows=na, color=stock_colors[0])
        results['n_hiv'] = init_res('Number living with HIV', color=stock_colors[0])
        results['n_hiv_by_age'] = init_res('Number living with HIV by age', n_rows=na, color=stock_colors[0])
        results['hiv_prevalence'] = init_res('HIV prevalence', color=stock_colors[0])
        results['female_hiv_prevalence'] = init_res('Female HIV prevalence', color=stock_colors[1])
        results['male_hiv_prevalence'] = init_res('Male HIV prevalence', color=stock_colors[2])
        results['hiv_prevalence_by_age'] = init_res('HIV prevalence by age', n_rows=na, color=stock_colors[0])
        results['hiv_deaths'] = init_res('New HIV deaths')
        results['hiv_deaths_by_age'] = init_res('New HIV deaths by age', n_rows=na, color=stock_colors[0])
        results['hiv_incidence'] = init_res('HIV incidence', color=stock_colors[0])
        results['hiv_incidence_by_age'] = init_res('HIV incidence by age', n_rows=na, color=stock_colors[0])
        results['n_hpv_by_age_with_hiv'] = init_res('Number HPV infections by age among HIV+', n_rows=na, color=stock_colors[0])
        results['n_hpv_by_age_no_hiv'] = init_res('Number HPV infections by age among HIV-', n_rows=na, color=stock_colors[0])
        results['hpv_prevalence_by_age_with_hiv'] = init_res('HPV prevalence by age among HIV+', n_rows=na, color=stock_colors[0])
        results['hpv_prevalence_by_age_no_hiv'] = init_res('HPV prevalence by age among HIV-', n_rows=na, color=stock_colors[1])
        results['cancers_by_age_with_hiv'] = init_res('Cancers by age among HIV+', n_rows=na, color=stock_colors[0])
        results['cancers_by_age_no_hiv'] = init_res('Cancers by age among HIV-', n_rows=na, color=stock_colors[1])
        results['cancers_with_hiv'] = init_res('Cancers among HIV+', color=stock_colors[0])
        results['cancers_no_hiv'] = init_res('Cancers among HIV-', color=stock_colors[1])
        results['cancer_incidence_with_hiv'] = init_res('Cancer incidence among HIV+', color=stock_colors[0])
        results['cancer_incidence_no_hiv'] = init_res('Cancer incidence among HIV-', color=stock_colors[1])
        results['cancer_incidence_by_age_with_hiv'] = init_res('Cancer incidence by age among HIV+', n_rows=na, color=stock_colors[0])
        results['cancer_incidence_by_age_no_hiv'] = init_res('Cancer incidence by age among HIV-', n_rows=na, color=stock_colors[1])
        results['n_females_with_hiv_alive_by_age'] = init_res('Number females with HIV alive by age', n_rows=na)
        results['n_females_no_hiv_alive_by_age'] = init_res('Number females without HIV alive by age', n_rows=na)
        results['n_females_with_hiv_alive'] = init_res('Number females with HIV alive')
        results['n_males_with_hiv_alive'] = init_res('Number males with HIV alive')
        results['n_females_no_hiv_alive'] = init_res('Number females without HIV alive')
        results['n_art'] = init_res('Number on ART')
        results['art_coverage'] = init_res('ART coverage')

        self.results = results

        return


    # %% HIV methods

    def set_hiv_prognoses(self, people, inds, year=None):
        ''' Set HIV outcomes '''

        art_coverage = self['art_coverage']  # Shorten
        shape = self['hiv_pars']['time_to_hiv_death_shape']
        dt = people.pars['dt']

        no_art_inds = hpu.true(people.hiv * ~people.art * people.alive)
        all_inds = np.concatenate((inds, no_art_inds))
        # t = people.t
        # update_freq = max(1, int(self['hiv_pars']['dt_art'] / dt))  # Ensure it's an integer not smaller than 1
        # if t % update_freq == 0:
        #
        # else:
        #     all_inds = sc.dcp(inds)

        if len(all_inds):
            # Extract index of current year
            all_years = np.array(list(art_coverage.keys()))
            year_ind = sc.findnearest(all_years, year)
            nearest_year = round(all_years[year_ind])

            # Apply ART coverage (being careful of level 0 and level 1 people!)
            art_cov = art_coverage[nearest_year]
            art_inds = np.empty(0)
            assign_dur_inds = np.empty(0)
            for sk in ['f', 'm']:
                art_year_sex = art_cov[sk]
                age_bins = art_year_sex[:, 0]
                this_art_cov = art_year_sex[:, 1]
                mf_inds = people.is_female if sk == 'f' else people.is_male
                mf_inds *= people.alive  # Only include people alive
                mf_art_inds = mf_inds * people.art
                mf_hiv_inds = mf_inds * people.hiv
                age_art_inds = np.digitize(people.age[mf_art_inds], age_bins)-1
                age_hiv_inds = np.digitize(people.age[mf_hiv_inds], age_bins)-1
                cur_n_age_bin = np.zeros(len(age_bins))
                cur_n_age_bin[age_art_inds] = people.scale_flows(hpu.true(mf_art_inds))
                desired_n_age_bin = np.zeros(len(age_bins))
                desired_n_age_bin[age_hiv_inds] = sc.randround(this_art_cov[age_hiv_inds] * people.scale_flows(hpu.true(mf_hiv_inds)))
                num_art_age_bin = desired_n_age_bin - cur_n_age_bin
                if np.sum(num_art_age_bin) > 0:
                    inds_age = np.digitize(people.age[all_inds], age_bins)-1
                    age_bins_to_fill = np.where(num_art_age_bin>0)[0]
                    for age_bin in age_bins_to_fill:
                        eligible_inds = all_inds[np.where(inds_age==age_bin)]
                        if len(eligible_inds):
                            art_probs = num_art_age_bin[age_bin] * people.scale[eligible_inds] / np.sum(people.scale[eligible_inds] ** 2)
                            art_inds = eligible_inds[hpu.true(hpu.binomial_arr(art_probs))]
                            people.art[art_inds] = True
                            people.date_art[art_inds] = people.t

                            # Get indices of people who are on ART who will not be virologically suppressed
                            art_failure_prob = self['hiv_pars']['art_failure_prob']
                            art_failure_probs = np.full(len(art_inds), fill_value=art_failure_prob, dtype=hpd.default_float)
                            art_failure_bools = hpu.binomial_arr(art_failure_probs)
                            art_failure_inds = art_inds[art_failure_bools]
                            people.art_failure[art_failure_inds] = True

                            # Get indices of those to assign durations for -- TODO, why not everyone?
                            assign_dur_inds = art_failure_inds  # Assign death to those not with ART failure

            no_art_inds = np.setdiff1d(inds, art_inds)
            assign_dur_inds = np.array(assign_dur_inds.tolist() + no_art_inds.tolist())

            if len(assign_dur_inds) > 0:
                scale = self['hiv_pars']['time_to_hiv_death_scale'](people.age[assign_dur_inds])
                scale = np.maximum(scale, 0)
                time_to_hiv_death = weibull_min.rvs(c=shape, scale=scale, size=len(assign_dur_inds))
                people.dur_hiv[assign_dur_inds] = time_to_hiv_death
                if self['hiv_pars']['model_hiv_death']:
                    people.date_dead_hiv[assign_dur_inds] = people.t + sc.randround(time_to_hiv_death / dt)

        return


    def check_hiv_death(self, people):
        '''
        Check for new deaths from HIV
        '''
        filter_inds = people.true('hiv')
        inds = people.check_inds(people.dead_hiv, people.date_dead_hiv, filter_inds=filter_inds)

        # Remove people and update flows
        people.remove_people(inds, cause='hiv')
        idx = int(people.t / self.resfreq)
        if len(inds):
            deaths_by_age = np.histogram(people.age[inds], bins=people.age_bin_edges, weights=people.scale[inds])[0]
            self.results['hiv_deaths'][idx] += people.scale_flows(inds)
            self.results['hiv_deaths_by_age'][:, idx] += deaths_by_age

        return


    def update_cd4(self, people):
        '''
        Update CD4 counts
        '''
        dt = people.pars['dt']
        filter_inds = people.true('hiv')
        if len(filter_inds):
            art_success_inds = filter_inds[hpu.true(people.art[filter_inds] & ~people.art_failure[filter_inds])]
            art_failure_inds = filter_inds[hpu.true(people.art[filter_inds] & people.art_failure[filter_inds])]
            not_art_inds = filter_inds[hpu.false(people.art[filter_inds])]
            cd4_decline_inds = np.concatenate((art_failure_inds, not_art_inds))

            # First take care of people unsuccessfully on ART or not on ART (CD4 will decline)
            cd4_remaining_inds = hpu.itrue(((people.t - people.date_hiv[cd4_decline_inds]) * dt) < people.dur_hiv[cd4_decline_inds], cd4_decline_inds)
            frac_prognosis = 100*((people.t - people.date_hiv[cd4_remaining_inds]) * dt) / people.dur_hiv[cd4_remaining_inds]
            cd4_change = self.cd4_decline_diff[frac_prognosis.astype(hpd.default_int)]
            people.cd4[cd4_remaining_inds] += cd4_change

            # Now take care of people successfully on ART (CD4 reconstitutes)
            mpy = 12
            months_on_ART = (people.t - people.date_art[art_success_inds]) * mpy
            cd4_change = self['hiv_pars']['cd4_reconstitution'](months_on_ART)
            cd4_change[cd4_change < 0] = 0
            people.cd4[art_success_inds] += cd4_change

        return


    def step(self, people=None, year=None):
        '''
        Wrapper method that checks for new HIV infections, updates prognoses, etc.
        '''

        new_infection_inds = self.new_hiv_infections(people, year) # Newly acquired HIV infections
        self.set_hiv_prognoses(people, new_infection_inds, year=year)  # Set ART adherence for those with HIV

        self.check_hiv_death(people)
        self.update_cd4(people)
        self.update_hpv_progs(people)
        self.update_hiv_results(people, new_infection_inds)

        return


    def new_hiv_infections(self, people, year=None):
        '''Apply HIV infection rates to population'''
        hiv_pars = self['infection_rates']
        all_years = np.array(list(hiv_pars.keys()))
        year_ind = sc.findnearest(all_years, year)
        nearest_year = all_years[year_ind]
        hiv_year = hiv_pars[nearest_year]
        dt = people.pars['dt']

        hiv_probs = np.zeros(len(people), dtype=hpd.default_float)
        for sk in ['f', 'm']:
            hiv_year_sex = hiv_year[sk]
            age_bins = hiv_year_sex[:, 0]
            hiv_rates = hiv_year_sex[:, 1] * dt
            mf_inds = people.is_female if sk == 'f' else people.is_male
            mf_inds *= people.alive  # Only include people alive
            age_inds = np.digitize(people.age[mf_inds], age_bins)
            hiv_probs[mf_inds] = hiv_rates[age_inds]
        hiv_probs[people.hiv] = 0  # not at risk if already infected

        # Get indices of people who acquire HIV
        hiv_inds = hpu.true(hpu.binomial_arr(hiv_probs))
        people.hiv[hiv_inds] = True
        people.cd4[hiv_inds] = hpu.sample(**self['hiv_pars']['cd4_start'], size=len(hiv_inds))
        people.date_hiv[hiv_inds] = people.t

        return hiv_inds


    def update_hpv_progs(self, people):
        ''' Update people's relative susceptibility, severity, and immunity '''

        hiv_inds = sc.autolist()

        for sn, cd4state in enumerate(self.cd4states):
            inds = sc.findinds((people.cd4 >= self.cd4_lb[sn]) & (people.cd4 < self.cd4_ub[sn]))
            hiv_inds += list(inds)
            if len(inds):
                for ir, rel_par in enumerate(['rel_sus', 'rel_sev', 'rel_imm']):
                    people[rel_par][inds] = self['hiv_pars'][rel_par][cd4state]

        # If anyone has HIV, update their HPV parameters
        if len(hiv_inds):

            hiv_inds = np.array(hiv_inds)

            dt = people.pars['dt']
            for g in range(people.pars['n_genotypes']):
                gpars = people.pars['genotype_pars'][g]
                hpv_inds = hpu.itruei((people.is_female & people.precin[g, :] & (np.isnan(people.date_cin[g,:]))), hiv_inds)  # Women with HIV who have pre-CIN and were not going to progress to CIN
                hpv_cin_inds = hpu.itruei((people.is_female & people.precin[g, :] & ~np.isnan(people.date_cin[g,:]) & (np.isnan(people.date_cancerous[g,:]))), hiv_inds)  # Women with HIV who have PRECIN and were going to develop CIN but not going to progress to cancer
                cin_inds = hpu.itruei((people.is_female & people.cin[g, :] & (np.isnan(people.date_cancerous[g,:]))), hiv_inds)  # Women with HIV who have CIN and were not going to progress to cancer
                if len(hpv_inds):  # Reevaluate these women's risk of developing CIN
                    people.set_prognoses(hpv_inds, g, gpars, dt)
                cin_reevaluate_inds = np.concatenate((hpv_cin_inds, cin_inds))
                if len(cin_reevaluate_inds):  # Reevaluate these women's risk of developing cancer
                    people.set_severity(cin_inds, g, gpars, dt)

        return


    def update_hiv_results(self, people, hiv_inds):
        ''' Update the HIV results '''

        idx = int(people.t / self.resfreq)

        #### Calculate flows
        # Flows get accumulated *every* time step
        self.results['hiv_infections'][idx] += people.scale_flows(hiv_inds)
        self.results['hiv_infections_by_age'][:, idx] += np.histogram(people.age[hiv_inds], bins=people.age_bin_edges, weights=people.scale[hiv_inds])[0]

        # Pull out those with cancer and HIV+
        cancer_today_inds = hpu.true(people.date_cancerous == people.t)
        if len(cancer_today_inds):
            hiv_bools = people.hiv[cancer_today_inds]
            cancer_today_hiv_pos_inds = cancer_today_inds[hiv_bools]
            cancer_today_hiv_neg_inds = cancer_today_inds[~hiv_bools]
            self.results['cancers_with_hiv'][idx] = people.scale_flows(cancer_today_hiv_pos_inds)
            self.results['cancers_no_hiv'][idx] = people.scale_flows(cancer_today_hiv_neg_inds)
            self.results['cancers_by_age_with_hiv'][:, idx] = \
            np.histogram(people.age[cancer_today_hiv_pos_inds], bins=people.age_bin_edges,
                         weights=people.scale[cancer_today_hiv_pos_inds])[0]
            self.results['cancers_by_age_no_hiv'][:, idx] = \
            np.histogram(people.age[cancer_today_hiv_neg_inds], bins=people.age_bin_edges,
                         weights=people.scale[cancer_today_hiv_neg_inds])[0]

        #### Calculate stocks
        # Stocks only get accumulated every nth time step, where n is the result frequency
        if people.t % self.resfreq == self.resfreq - 1:


            hivinds = hpu.true(people['hiv']* people.alive)
            self.results['n_hiv'][idx] = people.scale_flows(hivinds)
            self.results['n_hiv_by_age'][:, idx] = np.histogram(people.age[hivinds], bins=people.age_bin_edges, weights=people.scale[hivinds])[0]
            artinds = hpu.true(people.art*people.alive)
            self.results['n_art'][idx] = people.scale_flows(artinds)

            # Pull out those with HPV and HIV+
            hpvhivinds = hpu.true((people['hiv']) & people['infectious'])
            self.results['n_hpv_by_age_with_hiv'][:, idx] = np.histogram(people.age[hpvhivinds], bins=people.age_bin_edges, weights=people.scale[hpvhivinds])[0]

            # Pull out those with HPV and HIV-
            hpvnohivinds = hpu.true(~(people['hiv']) & people['infectious'])
            self.results['n_hpv_by_age_no_hiv'][:, idx] = np.histogram(people.age[hpvnohivinds], bins=people.age_bin_edges, weights=people.scale[hpvnohivinds])[0]

            alive_female_hiv_inds = hpu.true(people.alive*people.is_female*people.hiv)
            self.results['n_females_with_hiv_alive'][idx] = people.scale_flows(alive_female_hiv_inds)
            self.results['n_females_with_hiv_alive_by_age'][:, idx] = np.histogram(people.age[alive_female_hiv_inds], bins=people.age_bin_edges,
                 weights=people.scale[alive_female_hiv_inds])[0]
            alive_female_no_hiv_inds = hpu.true(people.alive * people.is_female * ~people.hiv)
            self.results['n_females_no_hiv_alive'][idx] = people.scale_flows(alive_female_no_hiv_inds)
            self.results['n_females_no_hiv_alive_by_age'][:, idx] = np.histogram(people.age[alive_female_no_hiv_inds], bins=people.age_bin_edges,
                 weights=people.scale[alive_female_no_hiv_inds])[0]

            alive_male_hiv_inds = hpu.true(people.alive*people.is_male*people.hiv)
            self.results['n_males_with_hiv_alive'][idx] = people.scale_flows(alive_male_hiv_inds)
        return


    def get_hiv_data(self, hiv_datafile=None, art_datafile=None):
        '''
        Load HIV incidence and art coverage data, if provided

        Args:
            location (str): must be provided if you want to run with HIV dynamics
            hiv_datafile (str):  must be provided if you want to run with HIV dynamics
            art_datafile (str):  must be provided if you want to run with HIV dynamics
            verbose (bool):  whether to print progress

        Returns:
            hiv_inc (dict): dictionary keyed by sex, storing arrays of HIV incidence over time by age
            art_cov (dict): dictionary keyed by sex, storing arrays of ART coverage over time by age
        '''

        if hiv_datafile is None and art_datafile is None:
            hiv_incidence_rates, hiv_mortality, art_coverage = None, None, None

        else:

            hiv_datafile = sc.promotetolist(hiv_datafile)
            art_datafile = sc.promotetolist(art_datafile)

            hiv_incidence_rates = dict()

            # Load data
            dfs_hiv = []
            for hiv_data in hiv_datafile:
                df_hiv = pd.read_csv(hiv_data)
                dfs_hiv.append(df_hiv)
            # df_inc = pd.read_csv(hiv_datafile)  # HIV incidence
            dfs_art = []
            for art_data in art_datafile:
                df_art = pd.read_csv(art_data)  # ART coverage
                dfs_art.append(df_art)

            # Process HIV and ART data
            sex_keys = ['Male', 'Female']
            sex_key_map = {'Male': 'm', 'Female': 'f'}

            hiv_mortality_dfs = []
            for id, df in enumerate(dfs_hiv):
                if 'mortality' in hiv_datafile[id]:
                    if 'female' in hiv_datafile[id]:
                        sex = 'f'
                    else:
                        sex = 'm'
                    df = df.set_index('age')
                    df = df.melt(ignore_index=False).reset_index()
                    df['Sex'] = sex
                    df.columns = ['Age', 'Year', 'Mortality', 'Sex']
                    hiv_mortality_dfs.append(df)
                else:

                    ## Start with incidence file
                    years = df['Year'].unique()

                    # Processing
                    for year in years:
                        hiv_incidence_rates[year] = dict()
                        for sk in sex_keys:
                            sk_out = sex_key_map[sk]
                            hiv_incidence_rates[year][sk_out] = np.concatenate(
                                [
                                    np.array([[9, 0]]),
                                    np.array(df[(df['Year'] == year) & (df['Sex'] == sk_out)][['Age', 'Incidence']],
                                             dtype=hpd.default_float),
                                    np.array([[150, 0]])  # Add another entry so that all older age groups are covered
                                ]
                            )
            hiv_mortality_df = pd.concat(hiv_mortality_dfs)
            hiv_mortality = dict()

            years = hiv_mortality_df['Year'].unique().astype(float)

            for year in years:
                year = round(year)
                hiv_mortality[year] = dict()
                for sk in sex_keys:
                    sk_out = sex_key_map[sk]
                    hiv_mortality[year][sk_out] = np.array(
                        hiv_mortality_df[(hiv_mortality_df['Year'] == str(year)) & (hiv_mortality_df['Sex'] == sk_out)][['Age', 'Mortality']],
                        dtype=hpd.default_float)

            # Now compute ART adherence over time/age
            art_dfs = []
            for id, df in enumerate(dfs_art):
                if 'females' in art_datafile[id]:
                    sex = 'f'
                else:
                    sex = 'm'
                df = df.set_index('age')
                df = df.melt(ignore_index=False).reset_index()
                df['Sex'] = sex
                df.columns = ['Age', 'Year', 'ART', 'Sex']
                art_dfs.append(df)

            art_df = pd.concat(art_dfs)
            art_coverage = dict()

            years = art_df['Year'].unique().astype(float)

            for year in years:
                year = round(year)
                art_coverage[year] = dict()
                for sk in sex_keys:
                    sk_out = sex_key_map[sk]
                    art_coverage[year][sk_out] = np.array(art_df[(art_df['Year'] == str(year)) & (art_df['Sex'] == sk_out)][['Age', 'ART']],
                                     dtype=hpd.default_float)

        return hiv_incidence_rates, hiv_mortality, art_coverage


    def load_data(self, hiv_datafile=None, art_datafile=None):
        ''' Load any data files that are used to create additional parameters, if provided '''
        hiv_data = sc.objdict()
        hiv_data.infection_rates, hiv_data.mortality_rates, hiv_data.art_coverage = self.get_hiv_data(hiv_datafile=hiv_datafile, art_datafile=art_datafile)
        return hiv_data


    def finalize(self, sim):
        '''
        Compute prevalence, incidence.
        '''
        res = self.results
        simres = sim.results


        # Compute HIV incidence and prevalence
        def safedivide(num, denom):
            ''' Define a variation on sc.safedivide that respects shape of numerator '''
            answer = np.zeros_like(num)
            fill_inds = (denom != 0).nonzero()
            if len(num.shape) == len(denom.shape):
                answer[fill_inds] = num[fill_inds] / denom[fill_inds]
            else:
                answer[:, fill_inds] = num[:, fill_inds] / denom[fill_inds]
            return answer

        alive_females = simres['n_alive_by_sex'][0, :]
        alive_males = simres['n_alive_by_sex'][1, :]

        ng = sim.pars['n_genotypes']
        no_hiv_by_age = simres['n_alive_by_age'][:] - res['n_hiv_by_age'][:]
        self.results['hiv_prevalence_by_age'][:] = safedivide(res['n_hiv_by_age'][:], simres['n_alive_by_age'][:])
        self.results['hiv_incidence'][:] = sc.safedivide(res['hiv_infections'][:], (simres['n_alive'][:] - res['n_hiv'][:]))
        self.results['hiv_incidence_by_age'][:] = sc.safedivide(res['hiv_infections_by_age'][:], (simres['n_alive_by_age'][:] - res['n_hiv_by_age'][:]))
        self.results['hiv_prevalence'][:] = (res['n_females_with_hiv_alive'][:] + res['n_males_with_hiv_alive'][:])/ simres['n_alive'][:]
        self.results['female_hiv_prevalence'][:] = res['n_females_with_hiv_alive'][:] / alive_females
        self.results['male_hiv_prevalence'][:] = res['n_males_with_hiv_alive'][:] / alive_males
        self.results['hpv_prevalence_by_age_with_hiv'][:] = safedivide(res['n_hpv_by_age_with_hiv'][:], ng*res['n_hiv_by_age'][:])
        self.results['hpv_prevalence_by_age_no_hiv'][:] = safedivide(res['n_hpv_by_age_no_hiv'][:], ng*no_hiv_by_age)
        self.results['art_coverage'][:] = safedivide(res['n_art'][:],res['n_hiv'][:])

        # Compute cancer incidence
        scale_factor = 1e5  # Cancer incidence are displayed as rates per 100k women
        self.results['cancer_incidence_with_hiv'][:] = safedivide(res['cancers_with_hiv'][:], res['n_females_with_hiv_alive'][:])*scale_factor
        self.results['cancer_incidence_no_hiv'][:] = safedivide(res['cancers_no_hiv'][:], res['n_females_no_hiv_alive'][:])*scale_factor
        self.results['cancer_incidence_by_age_with_hiv'][:] = safedivide(res['cancers_by_age_with_hiv'][:], res['n_females_with_hiv_alive_by_age'][:])*scale_factor
        self.results['cancer_incidence_by_age_no_hiv'][:] = safedivide(res['cancers_by_age_no_hiv'][:], res['n_females_no_hiv_alive_by_age'][:]) * scale_factor
        sim.results = sc.mergedicts(simres, self.results)
        return