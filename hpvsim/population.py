'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as hpu
from . import misc as hpm
# from . import base as cvb
from . import data as hpdata
from . import defaults as hpd
# from . import parameters as cvpar
from . import people as hpppl


# # Specify all externally visible functions this file defines
# __all__ = ['make_people', 'make_randpop', 'make_random_contacts']


def make_people(sim, popdict=None, die=True, reset=False, verbose=None, **kwargs):
    '''
    Make the people for the simulation.

    Usually called via ``sim.initialize()``.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict or People object
        die      (bool) : whether or not to fail if synthetic populations are requested but not available
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        verbose  (bool) : level of detail to print
        kwargs   (dict) : passed to make_randpop()

    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    network = sim['network'] # Shorten
    if verbose is None:
        verbose = sim['verbose']

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove

    if popdict is None:
        if network in ['random', 'basic']:
            popdict = make_randpop(sim, **kwargs) # Create a random network
        else: # pragma: no cover
            errormsg = f'Population type "{network}" not found; choices are random and others TBC'
            raise ValueError(errormsg)

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = hpppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], debut=popdict['debut'], contacts=popdict['contacts']) # List for storing the people

    sc.printv(f'Created {pop_size} people, average age {people.age.mean():0.2f} years', 2, verbose)

    return people


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys() # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or hp.People object, but instead is {type(popdict)}'
        raise TypeError(errormsg) from E

    # Check keys and lengths
    required_keys = ['uid', 'age', 'sex']
    popdict_keys = popdict.keys()
    pop_size = pars['pop_size']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != pop_size:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {pop_size}'
            raise ValueError(errormsg)

        isnan = np.isnan(popdict[key]).sum()
        if isnan:
            errormsg = f'Population not fully created: {isnan:,} NaNs found in {key}.'
            raise ValueError(errormsg)

    if ('contacts' not in popdict_keys) and (not hasattr(popdict, 'contacts')) and verbose:
        warnmsg = 'No contacts found. Please remember to add contacts before running the simulation.'
        hpm.warn(warnmsg)

    return


def make_randpop(pars, use_age_data=True, sex_ratio=0.5, microstructure='random', **kwargs):
    '''
    Make a random population, with contacts.

    This function returns a "popdict" dictionary, which has the following (required) keys:

        - uid: an array of (usually consecutive) integers of length N, uniquely identifying each agent
        - age: an array of floats of length N, the age in years of each agent
        - sex: an array of integers of length N (not currently used, so does not have to be binary)
        - contacts: list of length N listing the contacts; see make_random_contacts() for details
        - layer_keys: a list of strings representing the different contact layers in the population; see make_random_contacts() for details

    Args:
        pars (dict): the parameter dictionary or simulation object
        use_age_data (bool): whether to use location-specific age data
        use_household_data (bool): whether to use location-specific household size data
        sex_ratio (float): proportion of the population that is male (not currently used)
        microstructure (bool): whether or not to use the microstructuring algorithm to group contacts
        kwargs (dict): passed to contact creation method (e.g., make_hybrid_contacts)

    Returns:
        popdict (dict): a dictionary representing the population, with the following keys for a population of N agents with M contacts between them:
    '''

    pop_size = int(pars['pop_size']) # Number of people

    # Load age data and household demographics based on 2018 Seattle demographics by default, or country if available
    age_data = hpd.default_age_data
    location = pars['location']
    if location is not None:
        if pars['verbose']:
            print(f'Loading location-specific data for "{location}"')
        if use_age_data:
            try:
                age_data = hpdata.get_age_distribution(location)
            except ValueError as E:
                warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}), using default'
                hpm.warn(warnmsg)

    # Handle sexes, ages, and sexual debuts
    uids           = np.arange(pop_size, dtype=hpd.default_int)
    sexes          = np.random.binomial(1, sex_ratio, pop_size)
    age_data_min   = age_data[:,0]
    age_data_max   = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob   = age_data[:,2]
    age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
    age_bins        = hpu.n_multinomial(age_data_prob, pop_size) # Choose age bins
    ages            = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin
    debuts          = hpu.sample(**pars['debut'], size=pop_size)

    # Store output
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes
    popdict['debut'] = debuts

    # Deal with debuts and participation rates
    is_active = ages>debuts                 # Whether or not people have ever been sexually active
    active_inds = hpu.true(ages>debuts)     # Indices of sexually experienced people
    n_active = sum(is_active)               # Number of sexually experienced people

    # Create the contacts
    if microstructure == 'random':
        contacts = dict()
        for lkey,n in pars['partners'].items():
            n_active_layer = n_active*pars['layer_probs'][lkey]
            active_inds_layer = hpu.binomial_filter(pars['layer_probs'][lkey], active_inds)
            durations = pars['dur_pship'][lkey]
            contacts[lkey] = make_random_contacts(n_active_layer, n, durations, mapping=active_inds_layer, **kwargs)
    else: # pragma: no cover
        errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
        raise NotImplementedError(errormsg)

    popdict['contacts']   = contacts
    popdict['layer_keys'] = list(pars['partners'].keys())

    return popdict


def _tidy_edgelist(p1, p2, mapping):
    ''' Helper function to convert lists to arrays and optionally map arrays '''
    p1 = np.array(p1, dtype=hpd.default_int)
    p2 = np.array(p2, dtype=hpd.default_int)
    if mapping is not None:
        mapping = np.array(mapping, dtype=hpd.default_int)
        p1 = mapping[p1]
        p2 = mapping[p2]
    output = dict(p1=p1, p2=p2)
    return output


def make_random_contacts(pop_size, n, durations, overshoot=1.2, dispersion=None, mapping=None):
    '''
    Make random contacts for a single layer as an edgelist.

    Args:
        pop_size   (int)   : number of agents to create contacts between (N)
        n          (int) : the average number of contacts per person for this layer
        overshoot  (float) : to avoid needing to take multiple Poisson draws
        dispersion (float) : if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
        mapping    (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    '''

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Precalculate contacts
    n_all_contacts  = int(pop_size*n*overshoot) # The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value
    all_contacts    = hpu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
    if dispersion is None:
        p_count = hpu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = hpu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=hpd.default_int)

    # Make contacts
    count = 0
    for p in range(pop_size):
        n_contacts = p_count[p]
        these_contacts = all_contacts[count:count+n_contacts] # Assign people
        count += n_contacts
        p1.extend([p]*n_contacts)
        p2.extend(these_contacts)

    # Tidy up and add durations and start dates
    output = _tidy_edgelist(p1, p2, mapping)
    n_partnerships = len(output['p1'])
    output['dur'] = hpu.sample(**durations, size=n_partnerships)
    output['start'] = np.zeros(n_partnerships) # For now, assume commence at beginning of sim
    output['end'] = output['start'] + output['dur']

    return output

