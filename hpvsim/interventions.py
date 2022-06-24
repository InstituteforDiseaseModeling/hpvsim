'''
Specify the core interventions. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import sciris as sc
import pylab as pl
import inspect
from . import defaults as hpd
from . import parameters as hppar
from . import utils as hpu
from . import immunity as hpi
# from collections import defaultdict


#%% Helper functions

def find_day(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any day in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list/function): list of timepoints in the intervention, or a boolean array; or a function that returns these
        t (int): current simulation time (can be None if a boolean array is used)
        which (str): what to return: 'first', 'last', or 'all' indices
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable

    Returns:
        inds (list): list of matching timepoints; length zero or one unless which is 'all'

    New in version 2.1.2: arr can be a function with arguments interv and sim.
    '''
    if callable(arr):
        arr = arr(interv, sim)
        arr = sc.promotetoarray(arr)
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else: # pragma: no cover
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def get_subtargets(subtarget, sim):
    '''
    A small helper function to see if subtargeting is a list of indices to use,
    or a function that needs to be called. If a function, it must take a single
    argument, a sim object, and return a list of indices. Also validates the values.
    Currently designed for use with testing interventions, but could be generalized
    to other interventions. Not typically called directly by the user.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if callable(subtarget):
        subtarget = subtarget(sim)

    if 'inds' not in subtarget: # pragma: no cover
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']): # A function has been provided
        subtarget_inds = subtarget['inds'](sim) # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds'] # The indices are supplied directly

    # Validate the values
    if callable(subtarget['vals']): # A function has been provided
        subtarget_vals = subtarget['vals'](sim) # Call the function to get the indices
    else:
        subtarget_vals = subtarget['vals'] # The indices are supplied directly
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds): # pragma: no cover
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)

    return subtarget_inds, subtarget_vals

#%% Generic intervention classes

__all__ = ['Intervention']

class Intervention:
    '''
    Base class for interventions.
    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''
    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None):
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Screen"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.timepoints = [] # The start and end timepoints of the intervention
        self.initialized = False # Whether or not it has been initialized
        self.finalized = False # Whether or not it has been initialized
        return


    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = f'{type(self)} (error: {str(E)})' # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'


    def __call__(self, *args, **kwargs):
        # Makes Intervention(sim) equivalent to Intervention.apply(sim)
        if not self.initialized:  # pragma: no cover
            errormsg = f'Intervention (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        _,_,_,values = inspect.getargvalues(parent) # Get the values of the arguments
        if values:
            self.input_args = {}
            for key,value in values.items():
                if key == 'kwargs': # Store additional kwargs directly
                    for k2,v2 in value.items(): # pragma: no cover
                        self.input_args[k2] = v2 # These are already a dict
                elif key not in ['self', '__class__']: # Everything else, but skip these
                    self.input_args[key] = value
        return


    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized: # pragma: no cover
            raise RuntimeError('Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place: # pragma: no cover
            return self
        else:
            return sc.dcp(self)


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines at timepoints when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the timepoints stored in self.timepoints.
        However, if there is a self.plot_timepoints attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_timepoints'):
                timepoints = self.plot_timepoints
            else:
                timepoints = self.timepoints
            if sc.isiterable(timepoints):
                label_shown = False # Don't show the label more than once
                for timepoint in timepoints:
                    if sc.isnumber(timepoint):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        date = sim.yearvec[timepoint]
                        ax.axvline(date, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output



#%% Behavior change interventions
__all__ += ['dynamic_pars']

class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    You can also pass parameters to change directly as keyword arguments.

    Args:
        pars (dict): described above
        kwargs (dict): passed to Intervention()

    **Examples**::
        interv = hp.dynamic_pars(condoms=dict(timepoints=10, vals={'c':0.9})) # Increase condom use amount casual partners to 90%
        interv = hp.dynamic_pars({'beta':{'timepoints':[10, 15], 'vals':[0.005, 0.015]}, # At timepoint 10, reduce beta, then increase it again
                                  'debut':{'timepoints':10, 'vals':dict(f=dict(dist='normal', par1=20, par2=2.1), m=dict(dist='normal', par1=19.6, par2=1.8))}}) # Increase mean age of sexual debut
    '''

    def __init__(self, pars=None, **kwargs):

        # Find valid sim parameters and move matching keyword arguments to the pars dict
        pars = sc.mergedicts(pars) # Ensure it's a dictionary
        sim_par_keys = list(hppar.make_pars().keys()) # Get valid sim parameters
        kwarg_keys = [k for k in kwargs.keys() if k in sim_par_keys]
        for kkey in kwarg_keys:
            pars[kkey] = kwargs.pop(kkey)

        # Do standard initialization
        super().__init__(**kwargs) # Initialize the Intervention object

        # Handle the rest of the initialization
        subkeys = ['timepoints', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys(): # pragma: no cover
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise sc.KeyNotFoundError(errormsg)
                if sc.isnumber(pars[parkey][subkey]):
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
                else:
                    pars[parkey][subkey] = sc.promotetolist(pars[parkey][subkey])
            # timepoints = pars[parkey]['timepoints']
            # vals = pars[parkey]['vals']
            # if sc.isiterable(timepoints):
            #     len_timepoints = len(timepoints)
            #     len_vals = len(vals)
            #     if len_timepoints != len_vals:
            #         raise ValueError(f'Length of timepoints ({len_timepoints}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars

        return

    def initialize(self, sim):
        ''' Initialize with a sim '''
        for parkey in self.pars.keys():
            try: # First try to interpret the timepoints as dates
                tps = sim.get_t(self.pars[parkey]['timepoints'])  # Translate input to timepoints
            except:
                tps = []
                # See if it's in the time vector
                for tp in self.pars[parkey]['timepoints']:
                    if tp in sim.tvec:
                        tps.append(tp)
                    else: # Give up
                        errormsg = f'Could not parse timepoints provided for {parkey}.'
                        raise ValueError(errormsg)
            self.pars[parkey]['processed_timepoints'] = sc.promotetoarray(tps)
        self.initialized = True
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the timepoints, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            if t in parval['processed_timepoints']: # TODO: make this more robust
                self.timepoints.append(t)
                ind = sc.findinds(parval['processed_timepoints'], t)[0]
                val = parval['vals'][ind]
                if isinstance(val, dict):
                    sim[parkey].update(val) # Set the parameter if a nested dict
                else:
                    sim[parkey] = val # Set the parameter if not a dict
        return


__all__ += ['BaseVaccination', 'vaccinate_prob', 'Screening']

class BaseVaccination(Intervention):
    '''
    Apply a vaccine to a subset of the population.

    This base class implements the mechanism of vaccinating people to modify their immunity.
    It does not implement allocation of the vaccines, which is implemented by derived classes
    such as `cv.vaccinate`. The idea is that vaccination involves a series of standard operations
    to modify `cv.People` and applications will likely need to modify the vaccine parameters and
    test potentially complex allocation strategies. These should be accounted for by:

        - Custom vaccine parameters being passed in as a dictionary to the vaccine intervention
        - Custom vaccine allocations being implemented by a derived class overloading
          `BaseVaccination.select_people`. Any additional attributes required to manage the allocation
          can be defined in the derived class. Refer to `cv.vaccinate` or `cv.vaccinate_sequential` for
          an example of how to implement this.

    Some quantities are tracked during execution for reporting after running the simulation.
    These are:

        - ``doses``:             the number of vaccine doses per person

    Args:
        vaccine (dict/str) : which vaccine to use; see below for dict parameters
        label   (str)      : if vaccine is supplied as a dict, the name of the vaccine
        kwargs  (dict)     : passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        - ``imm_init``:  the initial immunity level (higher = more protection)
        - ``imm_boost``: how much of a boost being vaccinated on top of a previous dose or natural infection provides


    See ``parameters.py`` for additional examples of these parameters.


    '''
    def __init__(self, vaccine, label=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.index = None # Index of the vaccine in the sim; set later
        self.label = label # Vaccine label (used as a dict key)
        self.p     = None # Vaccine parameters
        self.doses = None # Record the number of doses given per person *by this intervention*
        self._parse_vaccine_pars(vaccine=vaccine) # Populate
        return


    def _parse_vaccine_pars(self, vaccine=None):
        ''' Unpack vaccine information, which may be given as a string or dict '''

        # Option 1: vaccines can be chosen from a list of pre-defined vaccines
        if isinstance(vaccine, str):

            choices, mapping = hppar.get_vaccine_choices()
            genotype_pars = hppar.get_vaccine_genotype_pars()
            dose_pars = hppar.get_vaccine_dose_pars()

            label = vaccine.lower()
            for txt in ['.', ' ', '&', '-', 'vaccine']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                vaccine_pars = sc.mergedicts(genotype_pars[label], dose_pars[label])
            else: # pragma: no cover
                errormsg = f'The selected vaccine "{vaccine}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

            if self.label is None:
                self.label = label

        # Option 2: vaccines can be specified as a dict of pars
        elif isinstance(vaccine, dict):

            # Parse label
            vaccine_pars = vaccine
            label = vaccine_pars.pop('label', None) # Allow including the label in the parameters
            if self.label is None: # pragma: no cover
                if label is None:
                    self.label = 'custom'
                else:
                    self.label = label

        else: # pragma: no cover
            errormsg = f'Could not understand {type(vaccine)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        # Set label and parameters
        self.p = sc.objdict(vaccine_pars)

        return


    def initialize(self, sim):
        super().initialize()

        # Populate any missing keys -- must be here, after genotypes are initialized
        default_genotype_pars   = hppar.get_vaccine_genotype_pars(default=True)
        default_dose_pars       = hppar.get_vaccine_dose_pars(default=True)
        genotype_labels         = list(sim['genotype_pars'].keys())
        dose_keys               = list(default_dose_pars.keys())

        # Handle dose keys
        for key in dose_keys:
            if key not in self.p:
                self.p[key] = default_dose_pars[key]

        # Handle genotypes
        for key in genotype_labels:
            if key not in self.p:
                if key in default_genotype_pars:
                    val = default_genotype_pars[key]
                else: # pragma: no cover
                    val = 1.0
                    if sim['verbose']: print(f'Note: No cross-immunity specified for vaccine {self.label} and genotype {key}, setting to 1.0')
                self.p[key] = val

        sim['vaccine_pars'][self.label] = self.p # Store the parameters
        self.index = list(sim['vaccine_pars'].keys()).index(self.label) # Find where we are in the list
        sim['vaccine_map'][self.index]  = self.label # Use that to populate the reverse mapping

        # Prepare to update sim['immunity']
        n_vax = self.index+1
        n_imm_sources = n_vax + len(sim['genotype_map'])
        immunity = sim['immunity']

        # add this vaccine to the immunity map
        sim['immunity_map'][n_imm_sources-1] = 'vaccine'
        if n_imm_sources > len(immunity): # need to add this vaccine, otherwise it's a duplicate
            vacc_mapping = [self.p[label] for label in sim['genotype_map'].values()]
            for _ in range(n_vax):
                vacc_mapping.append(1)
            vacc_mapping = np.reshape(vacc_mapping, (n_imm_sources, 1)).astype(hpd.default_float)
            immunity = np.hstack((immunity, vacc_mapping[0:len(immunity),]))
            immunity = np.vstack((immunity, np.transpose(vacc_mapping)))
            sim['immunity'] = immunity
            imm_boost = list(sim['imm_boost']) + [self.p['imm_boost']]
            sim['imm_boost'] = np.array(imm_boost)
            sim.people.set_pars(sim.pars)

        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def select_people(self, sim):
        """
        Return an array of indices of people to vaccinate
        Derived classes must implement this function to determine who to vaccinate at each timestep
        Args:
            sim: A cv.Sim instance
        Returns: Array of person indices
        """
        raise NotImplementedError


    def vaccinate(self, sim, vacc_inds, t=None):
        '''
        Vaccinate people

        This method applies the vaccine to the requested people indices. The indices of people vaccinated
        is returned. These may be different to the requested indices, because anyone that is dead will be
        skipped, as well as anyone already fully vaccinated (if booster=False). This could
        occur if a derived class does not filter out such people in its `select_people` method.

        Args:
            sim: A cv.Sim instance
            vacc_inds: An array of person indices to vaccinate
            t: Optionally override the day on which vaccinations are recorded for historical vaccination

        Returns: An array of person indices of people vaccinated
        '''

        if t is None:
            t = sim.t
        else: # pragma: no cover
            assert t <= sim.t, 'Overriding the vaccination day should only be used for historical vaccination' # High potential for errors to creep in if future vaccines could be scheduled here

        # Perform checks
        vacc_inds = vacc_inds[sim.people.alive[vacc_inds]] # Skip anyone that is dead
        # Skip anyone that has already had all the doses of *this* vaccine (not counting boosters).
        # Otherwise, they will receive the 2nd dose boost cumulatively for every subsequent dose.
        # Note, this does not preclude someone from getting additional doses of another vaccine (e.g. a booster)
        vacc_inds = vacc_inds[sim.people.doses[vacc_inds] < self.p['doses']]

        # Extract indices of already-vaccinated people and get indices of newly-vaccinated
        prior_vacc = hpu.true(sim.people.vaccinated)
        new_vacc   = np.setdiff1d(vacc_inds, prior_vacc)

        if len(vacc_inds):

            sim.people.vaccinated[vacc_inds] = True
            sim.people.vaccine_source[vacc_inds] = self.index
            sim.people.doses[vacc_inds] += 1
            sim.people.date_vaccinated[vacc_inds] = t
            imm_source = len(sim['genotype_map']) + self.index
            hpi.update_peak_immunity(sim.people, vacc_inds, self.p, imm_source, infection=False)

            factor = sim['pop_scale'] # Scale up by pop_scale, but then down by the current rescale_vec, which gets applied again when results are finalized TODO- not using rescale vec yet
            sim.people.flows['doses']      += len(vacc_inds)*factor # Count number of doses given
            sim.people.flows['vaccinated'] += len(new_vacc)*factor # Count number of people not already vaccinated given doses
            sim.people.total_flows['total_doses'] += len(vacc_inds)*factor
            sim.people.total_flows['total_vaccinated'] += len(new_vacc)*factor
        return vacc_inds


    def apply(self, sim):
        ''' Perform vaccination each timestep '''

        inds = self.select_people(sim)
        if len(inds):
            inds = self.vaccinate(sim, inds)
        return inds


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        obj.vaccinated = None
        obj.doses = None
        if hasattr(obj, 'second_dose_days'):
            obj.second_dose_days = None
        return obj


def check_doses(doses, interval):
    ''' Check that doses and intervals are supplied in correct formats '''

    # First check that they're both numbers
    if not sc.checktype(doses, int):
        raise ValueError(f'Doses must be an integer, not {doses}.')
    if interval is not None and not sc.isnumber(interval):
        errormsg = f"Can't understand the dosing interval given by '{interval}'. Dosing interval should be a number."
        raise ValueError(errormsg)

    # Now check that they're compatible
    if doses == 1 and interval is not None:
        raise ValueError("Can't use dosing intervals for vaccines with only one dose.")
    elif doses == 2 and interval is None:
        raise ValueError('Must specify a dosing interval if using a vaccine with more than one dose.')
    elif doses > 2:
        raise NotImplementedError('Scheduling three or more doses not yet supported; use a booster vaccine instead')

    return


def process_doses(num_doses, sim):
    ''' Handle different types of dose data'''
    if sc.isnumber(num_doses):
        num_people = num_doses
    elif callable(num_doses):
        num_people = num_doses(sim)
    elif sim.t in num_doses:
        num_people = num_doses[sim.t]
    else:
        num_people = 0
    return num_people


def process_sequence(sequence, sim):
    ''' Handle different types of prioritization sequence for vaccination '''
    if callable(sequence):
        sequence = sequence(sim.people)
    elif sequence == 'age':
        sequence = np.argsort(-sim.people.age)
    elif sequence is None:
        sequence = np.random.permutation(sim.n)
    elif sc.checktype(sequence, 'arraylike'):
        sequence = sc.promotetoarray(sequence)
    else:
        errormsg = f'Unable to interpret sequence {type(sequence)}: must be None, "age", callable, or an array'
        raise TypeError(errormsg)
    return sequence



class vaccinate_prob(BaseVaccination):
    '''
    Probability-based vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated.

    Args:
        vaccine (dict/str): which vaccine to use; see below for dict parameters
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        timepoints   (int/arr): the day or array of days to apply the interventions
        prob       (float): probability of being vaccinated (i.e., fraction of the population)
        subtarget   (dict): subtarget intervention to people with particular indices (see test_num() for details)
        kwargs      (dict): passed to Intervention()


    **Example**::

        pfizer = cv.vaccinate_prob(vaccine='pfizer', days=30, prob=0.7)
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    '''
    def __init__(self, vaccine, timepoints, label=None, prob=None, subtarget=None, **kwargs):
        super().__init__(vaccine,label=label,**kwargs) # Initialize the Intervention object
        self.tps      = sc.dcp(timepoints)
        if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
            prob = 1.0 if subtarget is None else 0.0
        self.prob      = prob
        self.subtarget = subtarget
        self.timepoints = timepoints
        self.dates = None  # Representations in terms of years, e.g. 2020.4, set during initialization
        self.second_dose_days = None  # Track scheduled second doses
        return


    def initialize(self, sim):
        super().initialize(sim)
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        self.second_dose_days     = [None]*sim.npts # People who get second dose (if relevant)
        check_doses(self.p['doses'], self.p['interval'])
        return


    def select_people(self, sim):

        vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose

        if sim.t >= np.min(self.timepoints):

            # Vaccinate people with their first dose
            for _ in find_day(self.timepoints, sim.t, interv=self, sim=sim):

                vacc_probs = np.zeros(len(sim.people))

                # Find eligible people
                vacc_probs[hpu.true(~sim.people.alive)] *= 0.0  # Do not vaccinate dead people
                eligible_inds = sc.findinds(~sim.people.vaccinated)
                vacc_probs[eligible_inds] = self.prob  # Assign equal vaccination probability to everyone

                # Apply any subtargeting
                if self.subtarget is not None:
                    subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                    vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted

                vacc_inds = hpu.true(hpu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated

                if len(vacc_inds):
                    if self.p.interval is not None:
                        # Schedule the doses
                        next_dose_days = sim.t + self.p.interval
                        if next_dose_days < sim['n_days']:
                            self.second_dose_days[next_dose_days] = vacc_inds

            # Also, if appropriate, vaccinate people with their second dose
            vacc_inds_dose2 = self.second_dose_days[sim.t]
            if vacc_inds_dose2 is not None:
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)

        return vacc_inds


class Screening(Intervention):
    '''
    Apply a screening program to a subset of the population.

    This base class implements the mechanism of screening people to identify and treat pre-cancerous lesions.
    Screening involves a series of standard operations to modify the trajectories of `hpv.People`. Screening algorithms
    can vary in complexity along the dimensions of primary screening modalities, triage modalities, treatment modalities,
    interval between screens and follow-up protocol, loss-to-follow-up, test characteristics, and efficacies.

    Args:
         primary_screen_test (dict/str)  : the screening test to use as a primary filtering method
         triage_screen_test  (dict/str)  : the screening test to use as a triage (or None)
         treatment           (dict/str)  : treatment to be used upon a positive test and/or triage
         screen_start_age    (int)       : age to start screening
         screen_interval     (int)       : interval between screens
         screen_stop_age     (int)       : age to stop screening
         timepoints          (int/arr)   : the day or array of days to apply the interventions
         prob                (float)     : probability of being screened (per screen)
         label               (str)       : the name of screening strategy
         kwargs (dict)      : passed to Intervention()

    If ``primary_screen_test`` and/or ``triage_screen_test`` is supplied as a dictionary, it must have the following parameters:
        - ``sensitivity``   : dictionary of probability of testing positive given each stage (i.e., HPV, CIN1, CIN2)
        - ``specificity``   : dictionary of specificity for each stage (i.e., HPV, CIN1, CIN2)

    If ``treatment`` is supplied as a dictionary, it must have the following parameters:
        - ``efficacy``   : dictionary of probability of clearing/regressing given stage

    '''

    def __init__(self, primary_screen_test, treatment, screen_start_age, screen_interval, screen_stop_age,
                 timepoints, prob=None, triage_screen_test=None, label=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.label = label  # Screening label (used as a dict key)
        self.p = None  # Screening parameters
        self.timepoints = timepoints
        if prob is None: # Populate default value of probability: 1
            prob = 1.0
        self.prob = prob
        self.screen_start_age = screen_start_age
        self.screen_interval = screen_interval
        self.screen_stop_age = screen_stop_age
        self._parse_screening_pars(screen=primary_screen_test)  # Populate
        self._parse_screening_pars(screen=triage_screen_test, triage=True)  # Populate
        self._parse_screening_pars(screen=treatment, treatment=True)  # Populate
        return

    def _parse_screening_pars(self, screen, triage=False, treatment=False):
        ''' Unpack screening information, which may be given as a string or dict '''

        # Option 1: screening can be chosen from a list of pre-defined screening strategies
        if isinstance(screen, str):

            if treatment:
                choices, mapping = hppar.get_treatment_choices()
                screen_pars = hppar.get_treatment_pars()
            else:
                choices, mapping = hppar.get_screen_choices()
                screen_pars = hppar.get_screen_pars()

            label = screen.lower()
            for txt in ['.', ' ', '&', '-', 'screen']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                screen_pars = screen_pars[label]
            else: # pragma: no cover
                errormsg = f'The selected screening method "{screen}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

            if self.label is None:
                self.label = label

        # Option 2: screening can be specified as a dict of pars
        elif isinstance(screen, dict):

            # Parse label
            screen_pars = screen
            label = screen_pars.pop('label', None) # Allow including the label in the parameters
            if self.label is None: # pragma: no cover
                if label is None:
                    self.label = 'custom'
                else:
                    self.label = label

        # Option 3: we are in triage and no triage is defined
        elif screen is None:
            screen_pars = None

        else: # pragma: no cover
            errormsg = f'Could not understand {type(screen)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        if triage:
            if screen is None:
                self.p = sc.mergedicts(self.p, {'triage': None})
            else:
                self.p = sc.mergedicts(self.p, {'triage': sc.objdict(screen_pars)})
        elif treatment:
            self.p = sc.mergedicts(self.p, {'treatment': sc.objdict(screen_pars)})
        else:
            # Set label and parameters
            self.p = {'primary': sc.objdict(screen_pars)}

        return


    def initialize(self, sim):
        super().initialize()
        self.timepoints, self.dates = sim.get_t(self.timepoints,return_date_format='str')  # Ensure timepoints and dates are in the right format
        self.validate_screen_pars(sim)
        self.p['screen_start_age'] = self.screen_start_age
        self.p['screen_interval'] = self.screen_interval
        self.p['screen_stop_age'] = self.screen_stop_age
        sim['screen_pars'][self.label] = self.p  # Store the parameters
        return

    def validate_screen_pars(self, sim):

        # pull out genotypes in sim to start the mapping process
        ng = sim['n_genotypes']
        genotype_map = sim['genotype_map']

        primary_screen_pars = self.p['primary']
        triage_screen_pars = self.p['triage']
        states = ['infectious', 'cin1', 'cin2', 'cin3']

        for state in states:
            tmp_screen_pars = np.ones(ng, dtype=hpd.default_float)
            for g in range(ng):
                tmp_screen_pars[g] = primary_screen_pars['sensitivity'][state][genotype_map[g]]
            self.p['primary']['sensitivity'][state] = tmp_screen_pars

        if triage_screen_pars is not None:
            tmp_screen_pars = np.ones(ng, dtype=hpd.default_float)
            for g in range(ng):
                tmp_screen_pars[g] = triage_screen_pars['sensitivity'][state][genotype_map[g]]
            self.p['triage']['sensitivity'][state] = tmp_screen_pars
        return

    def select_people(self, sim):
        """
        Return an array of indices of people to vaccinate
        Derived classes must implement this function to determine who to vaccinate at each timestep
        Args:
            sim: A cv.Sim instance
        Returns: Array of person indices
        """

        screen_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose
        if sim.t >= np.min(self.timepoints):
            screen_probs = np.zeros(len(sim.people))

            # Find people eligible for first screen
            eligible_inds = sc.findinds((sim.people.age >= self.p['screen_start_age']) &
                                        (sim.people.age <= self.p['screen_stop_age']) &
                                        (sim.people.screens == 0) )
            screen_probs[eligible_inds] = self.prob  # Assign equal screening probability to everyone

            # find people eligible for next screen
            next_eligible_inds = sc.findinds((sim.people.age >= self.p['screen_start_age']) &
                                    (sim.people.age <= self.p['screen_stop_age']) &
                                    (sim.people.date_screened == sim.t - self.p['screen_interval']))
            screen_probs[next_eligible_inds] = self.prob  # Assign equal screening probability to everyone

            screen_probs[hpu.true(~sim.people.alive)] *= 0.0  # Do not screen dead people
            screen_probs[hpu.true(~sim.people.is_male)] *= 0.0  # Do not screen men
            screen_inds = hpu.true(hpu.binomial_arr(screen_probs))  # Calculate who actually gets screened
        return screen_inds


    def screen(self, sim, screen_inds):
        '''
        Screen people

        This method applies the screening to the requested people indices. The indices of people screened
        is returned. These may be different to the requested indices, because anyone that is dead will be
        skipped.

        Args:
            sim: A cv.Sim instance
            screen_inds: An array of person indices to screen

        Returns: An array of person indices of people screened
        '''

        # Perform checks
        screen_inds = screen_inds[sim.people.alive[screen_inds]] # Skip anyone that is dead

        if len(screen_inds):

            sim.people.screened[screen_inds] = True
            sim.people.screens[screen_inds] += 1
            sim.people.date_screened[screen_inds] = sim.t

            # Do the actual screening!
            ng = sim['n_genotypes']
            dt = sim['dt']
            # Step 1, filter positives from primary screen
            primary_screen_pars = self.p['primary']['sensitivity']
            triage_screen_pars = self.p['triage']
            states = ['infectious', 'cin1', 'cin2', 'cin3']
            screen_pos = []

            for state in states:
                for g in range(ng):
                    screen_probs = np.zeros(len(sim.people))
                    tp_inds = hpu.true(sim.people[state][g,:])
                    screen_probs[tp_inds] = primary_screen_pars[state][g]
                    screen_pos_inds = hpu.true(hpu.binomial_arr(screen_probs))
                    screen_pos += list(screen_pos_inds)

            # remove duplicates from list
            screen_pos = np.array(list(set(screen_pos)))

            # Step 2, filter positives from triage (if appropriate) TODO: fill this part in
            if triage_screen_pars is not None:
                screen_pars = triage_screen_pars['sensitivity']


            # Step 3, treat and adjust prognoses accordingly
            treat_pars = self.p['treatment']['efficacy']
            treat_dur_pars = self.p['treatment']['time_to_clearance']

            # Find those with active CIN, if tx is efficacious, apply new time_to_clearance
            for state in ['cin1', 'cin2', 'cin3']:
                for g in range(ng):
                    inds = screen_pos[hpu.true(sim.people[state][g,screen_pos])]
                    eff_probs = np.zeros(len(inds))
                    eff_probs.fill(treat_pars[state])
                    eff_inds = hpu.true(hpu.binomial_arr(eff_probs))
                    eff_inds = inds[eff_inds]

                    dur_to_clearance = hpu.sample(**treat_dur_pars[state], size=len(eff_inds))
                    sim.people.date_clearance[g,eff_inds] = np.fmin(sim.people.date_clearance[g, eff_inds],sim.people.date_cin1[g, eff_inds] + np.ceil(dur_to_clearance / dt))

            factor = sim['pop_scale'] # Scale up by pop_scale, but then down by the current rescale_vec, which gets applied again when results are finalized TODO- not using rescale vec yet
        return screen_inds


    def apply(self, sim):
        ''' Perform vaccination each timestep '''

        inds = self.select_people(sim)
        if len(inds):
            inds = self.screen(sim, inds)
        return inds


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        return obj