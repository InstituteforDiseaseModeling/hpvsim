'''
Set the parameters for hpvsim.
'''

import numpy as np
import sciris as sc
from .settings import options as hpo # For setting global options
from . import misc as hpm
from . import defaults as hpd
from .data import loaders as hpdata

__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses']


def make_pars(set_prognoses=False, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = hp.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        version       (str):  if supplied, use parameters from this version
        kwargs        (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['pop_size']        = 20e3      # Number of agents
    pars['pop_scale']       = 1         # How much to scale the population
    pars['network']         = 'random'  # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location']        = None      # What location to load data from -- default Seattle
    pars['lx']              = None      # Proportion of people alive at the beginning of age interval x
    pars['birth_rates']     = None      # Birth rates, loaded below

    # Initialization parameters
    pars['init_hpv_prev']   = hpd.default_init_prev # Initial prevalence

    # Simulation parameters
    pars['start']           = 2015.         # Start of the simulation
    pars['end']             = None          # End of the simulation
    pars['n_years']         = 15            # Number of years to run, if end isn't specified. Note that this includes burn-in
    pars['burnin']          = 5             # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
    pars['dt']              = 0.2           # Timestep (in years)
    pars['rand_seed']       = 1             # Random seed, if None, don't reset
    pars['verbose']         = hpo.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
    pars['use_waning']      = False         # Whether or not to use waning immunity. If set to False, immunity from infection and vaccination is assumed to stay at the same level permanently

    # Network parameters, generally initialized after the population has been constructed
    pars['debut']           = dict(f=dict(dist='normal', par1=18.6, par2=2.1), # Location-specific data should be used here if possible
                                   m=dict(dist='normal', par1=19.6, par2=1.8))
    pars['partners']        = None  # The number of concurrent sexual partners for each partnership type
    pars['acts']            = None  # The number of sexual acts for each partnership type per year
    pars['condoms']         = None  # The proportion of acts in which condoms are used for each partnership type
    pars['layer_probs']     = None  # Proportion of the population in each partnership type
    pars['dur_pship']       = None  # Duration of partnerships in each partnership type
    pars['mixing']          = None  # Mixing matrices for storing age differences in partnerships
    pars['n_partner_types'] = 1  # Number of partnership types - reset below
    # pars['nonactive_by_age']= nonactive_by_age
    # pars['nonactive']       = None 

    # Basic disease transmission parameters
    pars['beta_dist']       = dict(dist='neg_binomial', par1=1.0, par2=1.0, step=0.01) # Distribution to draw individual level transmissibility TODO does this get used? if not remove.
    pars['beta']            = 0.05  # Per-act transmission probability; absolute value, calibrated
    pars['transf2m']        = 1.0   # Relative transmissibility of receptive partners in penile-vaginal intercourse; baseline value
    pars['transm2f']        = 3.69  # Relative transmissibility of insertive partners in penile-vaginal intercourse; based on https://doi.org/10.1038/srep10986: "For vaccination types, the risk of male-to-female transmission was higher than that of female-to-male transmission"

    # Probabilities of disease progression
    pars['rel_cin1_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cin2_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cin3_prob'] = 1.0  # Scale factor for proportion of CIN cases
    pars['rel_cancer_prob'] = 1.0  # Scale factor for proportion of CIN that develop into cancer
    pars['rel_death_prob'] = 1.0  # Scale factor for proportion of cancer cases that result in death
    pars['prognoses'] = None # Arrays of prognoses by duration; this is populated later

    # Parameters used to calculate immunity
    pars['imm_init'] = dict(dist='beta', par1=5, par2=3)  # beta distribution for initial level of immunity following infection clearance
    pars['imm_decay'] = dict(form=None)  # decay rate, with half life in years
    # pars['imm_decay'] = dict(form='exp_decay', init_val=1, half_life=20) # decay rate, with half life in years
    pars['imm_kin'] = None  # Constructed during sim initialization using the nab_decay parameters
    pars['imm_boost'] = []  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['immunity'] = None  # Matrix of immunity and cross-immunity factors, set by init_immunity() in immunity.py
    pars['immunity_map'] = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)

    # all genotype properties get populated by user in init_genotypes()
    pars['genotypes'] = []  # Genotypes of the virus; populated by the user below
    pars['genotype_map'] = dict()  # Reverse mapping from number to genotype key
    pars['genotype_pars'] = dict()  # Populated just below

    # Genotype parameters
    pars['n_genotypes'] = 1 # The number of genotypes circulating in the population
    pars['n_imm_sources'] = 1 # The number of immunity sources circulating in the population

    # Vaccine parameters
    pars['vaccine_pars'] = dict()  # Vaccines that are being used; populated during initialization
    pars['vaccine_map'] = dict()  # Reverse mapping from number to vaccine key

    # Screening parameters
    pars['screen_pars'] = dict()  # Screening method that is being used; populated during initialization

    # Duration of invasive cerival cancer before death
    pars['dur_cancer']  = dict(dist='lognormal', par1=6.0, par2=3.0)  # Duration of untreated cancer

    # Parameters determining relative transmissibility at each stage of disease
    pars['rel_trans'] = {}
    pars['rel_trans']['none']   = 1 # Baseline value
    pars['rel_trans']['cin1']   = 1 # Baseline value. Assumption, need data
    pars['rel_trans']['cin2']   = 0.2 # Assumption, need data
    pars['rel_trans']['cin3']   = 0.05 # Assumption, need data

    # Efficacy of protection
    pars['eff_condoms']     = 0.7  # The efficacy of condoms; https://www.nejm.org/doi/10.1056/NEJMoa053284?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

    # Events and interventions
    pars['interventions'] = []   # The interventions present in this simulation; populated by the user
    pars['analyzers']     = []   # Custom analysis functions; populated by the user
    pars['timelimit']     = None # Time limit for the simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses() # Default to duration-specific prognoses

    return pars


# Define which parameters need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['partners', 'mixing', 'acts', 'age_act_pars', 'layer_probs', 'dur_pship', 'condoms']


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Helper function to set layer-specific parameters. If layer keys are not provided,
    then set them based on the population type. This function is not usually called
    directly by the user, although it can sometimes be used to fix layer key mismatches
    (i.e. if the contact layers in the population do not match the parameters). More
    commonly, however, mismatches need to be fixed explicitly.

    Args:
        pars (dict): the parameters dictionary
        layer_keys (list): the layer keys of the population, if available
        force (bool): reset the parameters even if they already exist
    '''

    layer_defaults = {}
    # Specify defaults for random -- layer 'a' for 'all'
    layer_defaults['random'] = dict(
        partners    = dict(a=dict(dist='poisson', par1=0.01)), # Everyone in this layer has one partner; this captures *additional* partners. If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 partner
        acts        = dict(a=dict(dist='neg_binomial', par1=100,par2=50)),  # Default number of sexual acts per year for people at sexual peak
        age_act_pars = dict(a=dict(peak=35, retirement=75, debut_ratio=0.5, retirement_ratio=0.1)), # Parameters describing changes in coital frequency over agent lifespans
        layer_probs = dict(a=1.0),  # Default proportion of the population in each layer
        dur_pship   = dict(a=dict(dist='normal_pos', par1=5,par2=3)),    # Default duration of partnerships
        condoms     = dict(a=0.25),  # Default proportion of acts in which condoms are used
    )
    layer_defaults['random']['mixing'], layer_defaults['random']['layer_probs'] = get_mixing('random')

    # Specify defaults for basic sexual network with marital, casual, and one-off partners
    layer_defaults['default'] = dict(
        partners    = dict(m=dict(dist='poisson', par1=0.01), # Everyone in this layer has one marital partner; this captures *additional* marital partners. If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 spouse
                           c=dict(dist='poisson', par1=0.05), # If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 casual partner at a time
                           o=dict(dist='poisson', par1=0.0),), # If using a poisson distribution, par1 is roughly equal to the proportion of people with >1 one-off partner at a time. Can be set to zero since these relationships only last a single timestep
        acts         = dict(m=dict(dist='neg_binomial', par1=80, par2=40), # Default number of acts per year for people at sexual peak
                            c=dict(dist='neg_binomial', par1=10, par2=5), # Default number of acts per year for people at sexual peak
                            o=dict(dist='neg_binomial', par1=1,  par2=.01)),  # Default number of acts per year for people at sexual peak
        age_act_pars = dict(m=dict(peak=35, retirement=75, debut_ratio=0.5, retirement_ratio=0.1), # Parameters describing changes in coital frequency over agent lifespans
                            c=dict(peak=25, retirement=75, debut_ratio=0.5, retirement_ratio=0.1),
                            o=dict(peak=25, retirement=50, debut_ratio=0.5, retirement_ratio=0.1)),
        # layer_probs = dict(m=0.7, c=0.4, o=0.05),   # Default proportion of the population in each layer
        dur_pship   = dict(m=dict(dist='normal_pos', par1=10,par2=3),
                           c=dict(dist='normal_pos', par1=2, par2=1),
                           o=dict(dist='normal_pos', par1=0.1, par2=0.05)),
        condoms     = dict(m=0.01, c=0.5, o=0.6),  # Default proportion of acts in which condoms are used
    )
    layer_defaults['default']['mixing'], layer_defaults['default']['layer_probs'] = get_mixing('default')

    # Choose the parameter defaults based on the population type, and get the layer keys
    try:
        defaults = layer_defaults[pars['network']]
    except Exception as E:
        errormsg = f'Cannot load defaults for population type "{pars["network"]}"'
        raise ValueError(errormsg) from E
    default_layer_keys = list(defaults['acts'].keys()) # All layers should be the same, but use acts for convenience

    # Actually set the parameters
    for pkey in layer_pars:
        par = {} # Initialize this parameter
        default_val = layer_defaults['random'][pkey]['a'] # Get the default value for this parameter

        # If forcing, we overwrite any existing parameter values
        if force:
            par_dict = defaults[pkey] # Just use defaults
        else:
            par_dict = sc.mergedicts(defaults[pkey], pars.get(pkey, None)) # Use user-supplied parameters if available, else default

        # Figure out what the layer keys for this parameter are (may be different between parameters)
        if layer_keys:
            par_layer_keys = layer_keys # Use supplied layer keys
        else:
            par_layer_keys = list(sc.odict.fromkeys(default_layer_keys + list(par_dict.keys())))  # If not supplied, use the defaults, plus any extra from the par_dict; adapted from https://www.askpython.com/python/remove-duplicate-elements-from-list-python

        # Construct this parameter, layer by layer
        for lkey in par_layer_keys: # Loop over layers
            par[lkey] = par_dict.get(lkey, default_val) # Get the value for this layer if available, else use the default for random
        pars[pkey] = par # Save this parameter to the dictionary

    # Finally, update the number of partnership types
    pars['n_partner_types'] = len(par_layer_keys)

    return


def get_prognoses():
    '''
    Return the default parameter values for prognoses

    The prognosis probabilities are conditional given the previous disease state.

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities
    '''

    prognoses = dict(
        duration_cutoffs  = np.array([0,       1,          2,          3,          4]),      # Duration cutoffs (lower limits)
        seroconvert_probs = np.array([0.25,    0.5,        0.95,       1.0,        1.0]),    # Probability of seroconverting given duration of infection
        cin1_probs        = np.array([0.015,   0.15655,    0.30800,    0.50655,    0.70]),   # Conditional probability of developing CIN1 given HPV infection
        cin2_probs        = np.array([0.015,   0.0655,     0.3080,     0.60655,    0.90]),   # Conditional probability of developing CIN2 given CIN1
        cin3_probs        = np.array([0.15,    0.655,      0.80,       0.855,      0.90]),   # Conditional probability of developing CIN3 given CIN2
        cancer_probs      = np.array([0.0055,  0.0655,     0.2080,     0.50655,    0.90]),   # Conditional probability of developing cancer given CIN3
        )

    # Check that lengths match
    expected_len = len(prognoses['duration_cutoffs'])
    for key,val in prognoses.items():
        this_len = len(prognoses[key])
        if this_len != expected_len: # pragma: no cover
            errormsg = f'Lengths mismatch in prognoses: {expected_len} duration bins specified, but key "{key}" has {this_len} entries'
            raise ValueError(errormsg)

    return prognoses


def get_births_deaths(location=None, verbose=1, by_sex=True, overall=False, die=None):
    '''
    Get mortality and fertility data by location if provided, or use default

    Args:
        location (str):  location; if none specified, use default value for XXX
        verbose (bool):  whether to print progress
        by_sex   (bool): whether to get sex-specific death rates (default true)
        overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

    Returns:
        lx (dict): dictionary keyed by sex, storing arrays of lx - the number of people who survive to age x
        birth_rates (arr): array of crude birth rates by year
    '''

    birth_rates = hpd.default_birth_rates
    lx = hpd.default_lx
    if location is not None:
        if verbose:
            print(f'Loading location-specific demographic data for "{location}"')
        try:
            lx          = hpdata.get_death_rates(location=location, by_sex=by_sex, overall=overall)
            birth_rates = hpdata.get_birth_rates(location=location)
        except ValueError as E:
            warnmsg = f'Could not load demographic data for requested location "{location}" ({str(E)}), using default'
            hpm.warn(warnmsg, die=die)

    # Process the 85+ age group
    for sex in ['m','f']:
        if lx[sex][-1][0] == 85:
            last_val = lx[sex][-1][-1] # Save the last value
            lx[sex] = np.delete(lx[sex], -1, 0) # Remove the last row
            # Break this 15 year age bracket into 3x 5 year age brackets
            s85_89  = np.array([[85, 89, int(last_val*.7)]])
            s90_99  = np.array([[90, 99, int(last_val*.7*.5)]])
            s100    = np.array([[100, 110, 0]])
            lx[sex] = np.concatenate([lx[sex], s85_89, s90_99, s100])

    return birth_rates, lx



#%% Genotype/immunity parameters and functions

def get_genotype_choices():
    '''
    Define valid genotype names
    '''
    # List of choices available
    choices = {
        'hpv16':  ['hpv16', '16'],
        'hpv18': ['hpv18', '18'],
        'hpv6':  ['hpv6', '6'],
        'hpv11': ['hpv11', '11'],
        'hpv31': ['hpv31', '31'],
        'hpv33': ['hpv33', '33'],
        'hpv45': ['hpv45', '45'],
        'hpv52': ['hpv52', '52'],
        'hpv58': ['hpv58', '58'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping

def get_vaccine_choices():
    '''
    Define valid pre-defined vaccine names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'bivalent':  ['bivalent', 'hpv2', 'cervarix'],
        'quadrivalent': ['quadrivalent', 'hpv4', 'gardasil'],
        'nonavalent': ['nonavalent', 'hpv9', 'cervarix9'],
    }
    dose_1_options = ['1dose', '1doses', '1_dose', '1_doses', 'single_dose']
    dose_2_options = ['2dose', '2doses', '2_dose', '2_doses', 'double_dose']
    dose_3_options = ['3dose', '3doses', '3_dose', '3_doses', 'triple_dose']

    choices['bivalent_2dose'] = [f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_2_options]
    choices['bivalent_3dose'] = [f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_3_options]
    choices['bivalent'] = ['bivalent']+[f'{x}_{dose}' for x in choices['bivalent'] for dose in dose_1_options]
    choices['quadrivalent_2dose'] = [f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_2_options]
    choices['quadrivalent_3dose'] = [f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_3_options]
    choices['quadrivalent'] = ['quadrivalent']+[f'{x}_{dose}' for x in choices['quadrivalent'] for dose in dose_1_options]
    choices['nonavalent_2dose'] = [f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_2_options]
    choices['nonavalent_3dose'] = [f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_3_options]
    choices['nonavalent'] = ['nonavalent']+[f'{x}_{dose}' for x in choices['nonavalent'] for dose in dose_1_options]

    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping

def get_screen_choices():
    '''
    Define valid pre-defined screening names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'hpv':  ['hpv', 'hpvdna'],
        'hpv1618': ['hpv1618', 'hpvgenotyping'],
        'cytology': ['cytology', 'pap', 'papsmear'],
        'via': ['via', 'visualinspection'],
        'colposcopy': ['colposcopy', 'colpo'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping

def get_treatment_choices():
    '''
    Define valid pre-defined treatment names
    '''
    # List of choices currently available: new ones can be added to the list along with their aliases
    choices = {
        'default': ['default', None],
        'ablative':  ['ablative', 'thermal_ablation', 'TA'],
        'excisional': ['excisional'],
        'LEEP': ['LEEP'],
    }
    mapping = {name:key for key,synonyms in choices.items() for name in synonyms} # Flip from key:value to value:key
    return choices, mapping


def _get_from_pars(pars, default=False, key=None, defaultkey='default'):
    ''' Helper function to get the right output from genotype functions '''

    # If a string was provided, interpret it as a key and swap
    if isinstance(default, str):
        key, default = default, key

    # Handle output
    if key is not None:
        try:
            return pars[key]
        except Exception as E:
            errormsg = f'Key "{key}" not found; choices are: {sc.strjoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg) from E
    elif default:
        return pars[defaultkey]
    else:
        return pars


def get_genotype_pars(default=False, genotype=None):
    '''
    Define the default parameters for the different genotypes
    '''

    dur_dict = sc.objdict()
    for stage in ['none', 'cin1', 'cin2', 'cin3']:
        dur_dict[stage] = dict()

    pars = sc.objdict()

    pars.hpv16 = sc.objdict()
    pars.hpv16.dur = dict()
    pars.hpv16.dur['none']      = dict(dist='lognormal', par1=2.3625, par2=0.5)
                                    # Made the distribution wider to accommodate varying means
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv16.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv16.dur['cin2']      = dict(dist='gamma', par1=2.33, par2=6)
                                    # Shift this to the left compared to the cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv16.dur['cin3']      = dict(dist='gamma', par1=3.33, par2=6)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv16.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv16.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv16.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv16.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv16.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv16.imm_boost        = 1.0 # TODO: look for data

    pars.hpv18 = sc.objdict()
    pars.hpv18.dur = dict()
    pars.hpv18.dur['none']      = dict(dist='lognormal', par1=2.2483, par2=0.5)
                                    # Made the distribution wider to accommodate varying means
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv18.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv18.dur['cin2']      = dict(dist='gamma', par1=6.14, par2=2.49)
                                    # Shift this to the left compared to cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv18.dur['cin3']      = dict(dist='gamma', par1=9.14, par2=2.49)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv18.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv18.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv18.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv18.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv18.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv18.imm_boost        = 1.0 # TODO: look for data

    pars.hpv31 = sc.objdict()
    pars.hpv31.dur = dict()
    pars.hpv31.dur['none']      = dict(dist='lognormal', par1=2.5197, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv31.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv31.dur['cin2']      = dict(dist='gamma', par1=6.14, par2=2.49)
                                    # Shift this to the left compared to cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv31.dur['cin3']      = dict(dist='gamma', par1=9.14, par2=2.49)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv31.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv31.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv31.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv31.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv31.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv31.imm_boost        = 1.0 # TODO: look for data

    pars.hpv33 = sc.objdict()
    pars.hpv33.dur = dict()
    pars.hpv33.dur['none']      = dict(dist='lognormal', par1=2.3226, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv33.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv33.dur['cin2']      = dict(dist='gamma', par1=6.14, par2=2.49)
                                    # Shift this to the left compared to cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv33.dur['cin3']      = dict(dist='gamma', par1=9.14, par2=2.49)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv33.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv33.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv33.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv33.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv33.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv33.imm_boost        = 1.0 # TODO: look for data

    pars.hpv45 = sc.objdict()
    pars.hpv45.dur = dict()
    pars.hpv45.dur['none']      = dict(dist='lognormal', par1=2.0213, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv45.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv45.dur['cin2']      = dict(dist='gamma', par1=6.14, par2=2.49)
                                    # Shift this to the left compared to cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv45.dur['cin3']      = dict(dist='gamma', par1=9.14, par2=2.49)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv45.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv45.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv45.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv45.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv45.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv45.imm_boost        = 1.0 # TODO: look for data

    pars.hpv52 = sc.objdict()
    pars.hpv52.dur = dict()
    pars.hpv52.dur['none']      = dict(dist='lognormal', par1=2.3491, par2=1.0)
                                    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3707974/
                                    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.938&rep=rep1&type=pdf
                                    # https://academic.oup.com/jid/article/197/10/1436/2191990
    pars.hpv52.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv52.dur['cin2']      = dict(dist='gamma', par1=6.14, par2=2.49)
                                    # Shift this to the left compared to cin3 distribution
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv52.dur['cin3']      = dict(dist='gamma', par1=9.14, par2=2.49)
                                    # Assume that par1 = shape parameter, par2 = scale parameter
                                    # https://academic.oup.com/aje/article/178/7/1161/211254
    pars.hpv52.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv52.rel_cin1_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv52.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv52.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv52.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv52.imm_boost        = 1.0 # TODO: look for data

    pars.hpv6 = sc.objdict()
    pars.hpv6.dur = dict()
    pars.hpv6.dur['none']       = dict(dist='lognormal', par1=1.8245, par2=1.0)
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv6.dur['cin1']       = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv6.dur['cin2']       = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv6.dur['cin3']       = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv6.rel_beta          = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv6.rel_cin1_prob     = 0.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv6.rel_cin2_prob     = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv6.rel_cin3_prob     = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv6.rel_cancer_prob   = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv6.imm_boost         = 1.0 # TODO: look for data

    pars.hpv11 = sc.objdict()
    pars.hpv11.dur = dict()
    pars.hpv11.dur['none']      = dict(dist='lognormal', par1=1.8718, par2=1.0)
                                    # https://pubmed.ncbi.nlm.nih.gov/17416761/
    pars.hpv11.dur['cin1']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv11.dur['cin2']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv11.dur['cin3']      = dict(dist='lognormal', par1=2.0, par2=1.0) # PLACEHOLDERS; INSERT SOURCE
    pars.hpv11.rel_beta         = 1.0 # Transmission was relatively homogeneous across HPV genotypes, alpha species, and oncogenic risk categories -- doi: 10.2196/11284
    pars.hpv11.rel_cin1_prob    = 0.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv11.rel_cin2_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv11.rel_cin3_prob    = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv11.rel_cancer_prob  = 1.0 # Set this value to zero for non-carcinogenic genotypes
    pars.hpv11.imm_boost        = 1.0 # TODO: look for data

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_cross_immunity(default=False, genotype=None):
    '''
    Get the cross immunity between each genotype in a sim
    '''
    pars = dict(

        hpv16 = dict(
            hpv16  = 1.0, # Default for own-immunity
            hpv18 = 0, # Assumption
            hpv31  = 0, # Assumption
            hpv33 = 0, # Assumption
            hpv45 = 0, # Assumption
            hpv52 = 0, # Assumption
            hpv58 = 0, # Assumption
            hpv6 = 0, # Assumption
            hpv11 = 0, # Assumption
        ),

        hpv18 = dict(
            hpv16=0,  # Default for own-immunity
            hpv18=1.0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        hpv31=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=1.0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        hpv33=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=1.0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        hpv45=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=1.0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        hpv52=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=1.0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption

        ),

        hpv58=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=1.0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption

        ),

        hpv6=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=1.0,  # Assumption
            hpv11=0,  # Assumption

        ),

        hpv11=dict(
            hpv16=0,  # Default for own-immunity
            hpv18=0,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=1.0,  # Assumption

        ),


    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')


def get_mixing(network=None):
    '''
    Define defaults for sexual mixing matrices and the proportion of people of each age group
    who have relationships of each type.

    The mixing matrices represent males in the rows and females in the columns.
    Non-zero entires mean that there are some relationships between males/females of the age
    bands in the row/column combination. Entries >1 or <1 can be used to represent relative
    likelihoods of males of a given age cohort partnering with females of that cohort.
    For example, a mixing matrix like the following would mean that males aged 15-30 were twice
    likely to partner with females of age 15-30 compared to females aged 30-50.
        mixing = np.array([
                                #15, 30,
                            [15,  2, 1],
                            [30,  1, 1]])
    Note that the first column of the mixing matrix represents the age bins. The same age bins
    must be used for males and females, i.e. the matrix must be square.

    The proportion of people of each age group who have relationships of each type is
    given by the layer_probs array. The first row represents the age bins, the second row
    represents the proportion of females of each age who have relationships of each type, and
    the third row represents the proportion of males of each age who have relationships of
    each type.
    '''

    if network == 'default':

        mixing = dict(
            m=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5 ,.1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ]),
            c=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ]),
            o=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ]),
        )

        layer_probs = dict(
            m=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.04,   0.2,  0.6,  0.8,  0.8,  0.8,  0.75,  0.65,  0.55,  0.4,  0.4,  0.4,  0.4,  0.4], # Share of females of each age who are married
                [ 0,  0,  0.01,  0.01,  0.2,  0.6,  0.8,  0.9,  0.90,  0.90,  0.90,  0.8,  0.7,  0.6,  0.5,  0.6]] # Share of males of each age who are married
            ),
            c=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.10,   0.6,  0.3,  0.1,  0.1,  0.1,   0.1,  0.05,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # Share of females of each age having casual relationships
                [ 0,  0,  0.05,   0.5,  0.5,  0.3,  0.4,  0.5,   0.5,   0.4,   0.3,  0.1, 0.05, 0.01, 0.01, 0.01]], # Share of males of each age having casual relationships
            ),
            o=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.01,  0.05, 0.05, 0.04, 0.03, 0.02,  0.01,  0.01,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # Share of females of each age having one-off relationships
                [ 0,  0,  0.01,  0.01, 0.01, 0.02, 0.03, 0.04,  0.05,  0.05,  0.03, 0.02, 0.01, 0.01, 0.01, 0.01]], # Share of males of each age having one-off relationships
            ),
        )

    elif network == 'random':
        mixing = dict(
            a=np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  0,  0,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0,  0,  0,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
        ])
        )
        layer_probs = dict(
            a=np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,   55,   60,   65,   70,   75],
                [ 0,  0,  0.04,   0.2,  0.6,  0.8,  0.8,  0.8,  0.75,  0.65,  0.55,  0.4,  0.4,  0.4,  0.4,  0.4], # Share of females of each age who are married
                [ 0,  0,  0.01,  0.01,  0.2,  0.6,  0.8,  0.9,  0.90,  0.90,  0.90,  0.8,  0.7,  0.6,  0.5,  0.6]] # Share of males of each age who are married
            ))

    else:
        errormsg = f'Network "{network}" not found; the choices at this stage are random and default.'
        raise ValueError(errormsg)

    return mixing, layer_probs


def get_vaccine_genotype_pars(default=False, vaccine=None):
    '''
    Define the cross-immunity of each vaccine against each genotype
    '''
    pars = dict(

        default = dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        bivalent = dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=0,  # Assumption
            hpv11=0,  # Assumption
        ),

        quadrivalent=dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=0,  # Assumption
            hpv33=0,  # Assumption
            hpv45=0,  # Assumption
            hpv52=0,  # Assumption
            hpv58=0,  # Assumption
            hpv6=1,  # Assumption
            hpv11=1,  # Assumption
        ),

        nonavalent=dict(
            hpv16=1,
            hpv18=1,  # Assumption
            hpv31=1,  # Assumption
            hpv33=1,  # Assumption
            hpv45=1,  # Assumption
            hpv52=1,  # Assumption
            hpv58=1,  # Assumption
            hpv6=1,  # Assumption
            hpv11=1,  # Assumption
        ),
    )

    pars['bivalent_2dose'] = pars['bivalent']
    pars['bivalent_3dose'] = pars['bivalent']
    pars['quadrivalent_2dose'] = pars['quadrivalent']
    pars['quadrivalent_3dose'] = pars['quadrivalent']
    pars['nonavalent_2dose'] = pars['nonavalent']
    pars['nonavalent_3dose'] = pars['nonavalent']

    return _get_from_pars(pars, default=default, key=vaccine)


def get_vaccine_dose_pars(default=False, vaccine=None):
    '''
    Define the parameters for each vaccine
    '''

    pars = dict(

        default = dict(
            imm_init  = dict(dist='beta', par1=30, par2=2), # Initial distribution of immunity
            doses     = 1, # Number of doses for this vaccine
            interval  = None, # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent_2dose = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=2,  # Number of doses for this vaccine
            interval=0.5,  # Interval between doses in years
            imm_boost=1.2,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        bivalent_3dose = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=3,  # Number of doses for this vaccine
            interval=[0.2, 0.5],  # Interval between doses in years
            imm_boost=[1.2, 1.1],  # Factor by which each dose increases immunity
        ),

        quadrivalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),

        nonavalent = dict(
            imm_init=dict(dist='beta', par1=30, par2=2),  # Initial distribution of immunity
            doses=1,  # Number of doses for this vaccine
            interval=None,  # Interval between doses
            imm_boost=None,  # For vaccines wiht >1 dose, the factor by which each additional boost increases immunity
        ),
    )

    return _get_from_pars(pars, default, key=vaccine)


def get_screen_pars(screen=None):
    '''
    Define the parameters for each screen method
    should extract test positivity, inadequacy
    '''

    pars = dict(
        hpv = dict(
            test_positivity=dict(
                infectious=0.56,
                cin1=0.8415,
                cin2=0.93,
                cin3=0.984,
                cancerous=0.984,
            ),
            inadequacy=0,
        ),

        hpv1618 = dict(
            test_positivity=dict(
                infectious=0.56,
                cin1=0.8415,
                cin2=0.93,
                cin3=0.984,
                cancerous=0.984,
            ),
            inadequacy=0,
        ),

        cytology = dict(
            test_positivity=dict(
                infectious=0.11,
                cin1=0.5405,
                cin2=0.5941,
                cin3=0.7551,
                cancerous=0.984,
            ),
            inadequacy=0.07,
        ),

        via=dict(
            test_positivity=dict(
                infectious=0.25,
                cin1=0.3,
                cin2=0.45,
                cin3=0.41,
                cancerous=0.6,
            ),
            inadequacy=0,
        ),
    )

    return _get_from_pars(pars, key=screen)

def get_treatment_pars(screen=None):
    '''
    Define the parameters for each treatment method
    '''

    pars = dict(
        excisional = dict(
            efficacy=dict(
                cin2=1,
                cin3=1,
            ),
            persistence=dict(
                hpv16=0.66, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv18=0.66, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv31=0.524, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv33=0.524, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv45=0.524, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv52=0.524, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv58=0.524, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv6=0.25, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv11=0.25, # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
            )
        ),

        ablative=dict(
            efficacy=dict(
                cin1=0.81,
                cin2=0.81,
                cin3=0.81,
            ),
            persistence=dict(
                hpv16=0.66,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv18=0.66,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv31=0.524,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv33=0.524,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv45=0.524,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv52=0.524,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv58=0.524,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv6=0.25,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
                hpv11=0.25,  # https://obgyn.onlinelibrary.wiley.com/doi/10.1111/jog.12196
            )
        ),
    )

    return _get_from_pars(pars, key=screen)
