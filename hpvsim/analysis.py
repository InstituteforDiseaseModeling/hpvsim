'''
Additional analysis functions that are not part of the core workflow,
but which are useful for particular investigations.
'''

import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import utils as hpu
from . import misc as hpm
from . import interventions as hpi
from . import plotting as hppl
from . import defaults as hpd
from . import parameters as hppar
# from . import run as cvr
from .settings import options as hpo # For setting global options
import seaborn as sns


__all__ = ['Analyzer', 'snapshot', 'age_pyramid', 'age_results', 'Calibration']


class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class. Analyzers are used
    to provide more detailed information about a simulation than is available by
    default -- for example, pulling states out of sim.people on a particular timestep
    before it gets updated in the next timestep.

    To retrieve a particular analyzer from a sim, use sim.get_analyzer().

    Args:
        label (str): a label for the Analyzer (used for ease of identification)
    '''

    def __init__(self, label=None):
        if label is None:
            label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Record ages"
        self.initialized = False
        self.finalized = False
        return


    def __call__(self, *args, **kwargs):
        # Makes Analyzer(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def initialize(self, sim=None):
        '''
        Initialize the analyzer, e.g. convert date strings to integers.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize analyzer

        This method is run once as part of `sim.finalize()` enabling the analyzer to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Analyzer already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply analyzer at each time point. The analyzer has full access to the
        sim object, and typically stores data/results in itself. This is the core
        method which each analyzer object needs to implement.

        Args:
            sim: the Sim instance
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:
            return self
        else:
            return sc.dcp(self)


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. This method will attempt to JSONify each attribute of the
        intervention, skipping any that fail.

        Returns:
            JSON-serializable representation
        '''
        # Set the name
        json = {}
        json['analyzer_name'] = self.label if hasattr(self, 'label') else None
        json['analyzer_class'] = self.__class__.__name__

        # Loop over the attributes and try to process
        attrs = self.__dict__.keys()
        for attr in attrs:
            try:
                data = getattr(self, attr)
                try:
                    attjson = sc.jsonify(data)
                    json[attr] = attjson
                except Exception as E:
                    json[attr] = f'Could not jsonify "{attr}" ({type(data)}): "{str(E)}"'
            except Exception as E2:
                json[attr] = f'Could not jsonify "{attr}": "{str(E2)}"'
        return json


def validate_recorded_dates(sim, requested_dates, recorded_dates, die=True):
    '''
    Helper method to ensure that dates recorded by an analyzer match the ones
    requested.
    '''
    requested_dates = sorted(list(requested_dates))
    recorded_dates = sorted(list(recorded_dates))
    if recorded_dates != requested_dates: # pragma: no cover
        errormsg = f'The dates {requested_dates} were requested but only {recorded_dates} were recorded: please check the dates fall between {sim["start"]} and {sim["end"]} and the sim was actually run'
        if die:
            raise RuntimeError(errormsg)
        else:
            print(errormsg)
    return



class snapshot(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method.

    Args:
        timepoints  (list): list of ints/strings/date objects, the days on which to take the snapshot
        die         (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs      (dict): passed to Analyzer()


    **Example**::

        sim = cv.Sim(analyzers=cv.snapshot('2015.4', '2020'))
        sim.run()
        snapshot = sim['analyzers'][0]
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020']       # Option 2
        people = snapshot.get('2020')             # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    '''

    def __init__(self, timepoints, *args, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timepoints = sc.promotetolist(timepoints) # Combine multiple timepoints
        self.timepoints     = timepoints
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.snapshots      = sc.odict() # Store the actual snapshots
        return


    def initialize(self, sim):
        self.start = sim['start'] # Store the simulation start
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        self.initialized = True
        return


    def apply(self, sim):
        for ind in sc.findinds(self.timepoints, sim.t):
            date = self.dates[ind]
            self.snapshots[date] = sc.dcp(sim.people) # Take snapshot!


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.snapshots.keys(), die=self.die)
        return


    def get(self, key=None):
        ''' Retrieve a snapshot from the given key (int, str, or date) '''
        if key is None:
            key = self.dates[0]
        date = key # TODO: consider ways to make this more robust
        if date in self.snapshots:
            snapshot = self.snapshots[date]
        else: 
            dates = ', '.join(list(self.snapshots.keys()))
            errormsg = f'Could not find snapshot date {date}: choices are {self.dates}'
            raise sc.KeyNotFoundError(errormsg)
        return snapshot



class age_pyramid(Analyzer):
    '''
    Constructs an age/sex pyramid at specified points within the sim. Can be used with data
    Args:
        timepoints  (list): list of ints/strings/date objects, the days on which to take the snapshot
        die         (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs      (dict): passed to Analyzer()


    **Example**::
        sim = cv.Sim(analyzers=hp.age_pyramid('2015', '2020'))
        sim.run()
        age_pyramid = sim['analyzers'][0]
    '''

    def __init__(self, timepoints, edges=None, age_labels=None, datafile=None, die=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timepoints          = sc.promotetolist(timepoints) # Combine multiple timepoints
        self.timepoints     = timepoints
        self.edges          = edges # Edges of bins
        self.datafile       = datafile # Data file to load
        self.bins           = None # Age bins, calculated from edges
        self.data           = None # Store the loaded data
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.age_labels     = age_labels # Labels for the age bins - will be automatically generated if not provided
        self.age_pyramids   = sc.odict() # Store the age pyramids
        return


    def initialize(self, sim):
        super().initialize()

        # Handle timepoints and dates
        self.start = sim['start']   # Store the simulation start
        self.end   = sim['end']     # Store simulation end
        if self.timepoints is None:
            self.timepoints = self.end # If no day is supplied, use the last day
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        max_hist_time = self.timepoints[-1]
        max_sim_time = sim['end']
        if max_hist_time > max_sim_time:
            errormsg = f'Cannot create histogram for {self.dates[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
            raise ValueError(errormsg)

        # Handle edges, age bins, and labels
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins
        if self.age_labels is None:
            self.age_labels = [f'{int(self.edges[i])}-{int(self.edges[i+1])}' for i in range(len(self.edges)-1)]
            self.age_labels.append(f'{int(self.edges[-1])}+')

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = hpm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

            # Validate the data. Currently we only allow the same timepoints and age brackets
            data_dates = {str(float(i)) for i in self.data.year}
            if len(set(self.dates)-data_dates) or len(data_dates-set(self.dates)):
                string = f'Dates provided in the age pyramid datafile ({data_dates}) are not the same as the age pyramid dates that were requested ({self.dates}).'
                if self.die:
                    raise ValueError(string)
                else:
                    string += '\nPlots will only show requested dates, not all dates in the datafile.'
                    print(string)
            self.data_dates = data_dates

            # Validate the edges - must be the same as requested edges from the model output
            data_edges = np.array(self.data.age.unique(), dtype=float)
            if not np.array_equal(np.sort(self.edges),np.sort(data_edges)):
                errormsg = f'Age bins provided in the age pyramid datafile ({data_edges}) are not the same as the age pyramid age bins that were requested ({self.edges}).'
                raise ValueError(errormsg)

        self.initialized = True

        return


    def apply(self, sim):
        for ind in sc.findinds(self.timepoints, sim.t):

            date = self.dates[ind]
            self.age_pyramids[date] = sc.objdict() # Initialize the dictionary
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor
            age = sim.people.age # Get the age distribution
            self.age_pyramids[date]['bins'] = self.bins # Copy here for convenience
            for sb,sex in enumerate(['m','f']): # Loop over each sex; sb stands for sex boolean, translating the labels to 0/1
                inds = (sim.people.alive*(sim.people.sex==sb)).nonzero()[0]
                self.age_pyramids[date][sex] = np.histogram(age[inds], bins=self.edges)[0]*scale  # Bin people


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.age_pyramids.keys(), die=self.die)
        return


    def plot(self, m_color='#4682b4', f_color='#ee7989', fig_args=None, axis_args=None, data_args=None,
             percentages=True, do_save=None, fig_path=None, do_show=True, **kwargs):
        '''
        Plot the age pyramids

        Args:
            m_color (hex or rgb): the color of the bars for males
            f_color (hex or rgb): the color of the bars for females
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            percentages (bool): whether to plot the pyramid as percentages or numbers
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hp.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        all_args = sc.mergedicts(fig_args, axis_args, d_args)

        # Initialize
        fig = pl.figure(**fig_args)
        labels = list(reversed(self.age_labels))

        # Set properties depending on data
        if self.data is None: # Simple case: just plot model output
            n_plots = len(self.timepoints)
            n_rows, n_cols = sc.get_rows_cols(n_plots)
        else: # Complex case: add data
            n_cols = 2 # Data plots go in the right column
            n_rows = len(self.timepoints) # We only show plots for requested timepoints

        # Handle windows and what to plot
        pyramidsdict = self.age_pyramids
        if not len(pyramidsdict):
            errormsg = f'Cannot plot since no age pyramids were recorded (scheduled timepoints: {self.timepoints})'
            raise ValueError(errormsg)

        # Make the figure(s)
        xlabel = 'Share of population by sex' if percentages else 'Population by sex'
        with hpo.with_style(**kwargs):
            count=1
            for date,pyramid in pyramidsdict.items():
                pl.subplots_adjust(**axis_args)
                bins = pyramid['bins']

                # Prepare data
                pydf = pd.DataFrame(pyramid)
                if percentages:
                    pydf['m'] = pydf['m'] / sum(pydf['m'])
                    pydf['f'] = pydf['f'] / sum(pydf['f'])
                pydf['f']=-pydf['f'] # Reverse values for females to get on same axis

                # Start making plot
                ax = pl.subplot(n_rows, n_cols, count)
                sns.barplot(x='m', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
                sns.barplot(x='f', y='bins', data=pydf, order=np.flip(bins), orient='h', ax=ax, color=f_color)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Age group')
                ax.set_yticklabels(labels[1:])
                xticks = ax.get_xticks()
                if percentages:
                    xlabels = [f'{abs(i):.2f}' for i in xticks]
                else:
                    xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.set_title(f'{date}')
                count +=1

                if self.data is not None:

                    if date in self.data_dates:
                        datadf = self.data[self.data.year==float(date)]
                        # Consistent naming of males and females
                        datadf.columns = datadf.columns.str[0]
                        datadf.columns = datadf.columns.str.lower()
                        if percentages:
                            datadf = datadf.assign(m=datadf['m'] / sum(datadf['m']), f=datadf['f'] / sum(datadf['f']))
                        datadf = datadf.assign(f=-datadf['f'])

                        # Start making plot
                        ax = pl.subplot(n_rows, n_cols, count)
                        sns.barplot(x='m', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color=m_color)
                        sns.barplot(x='f', y='a', data=datadf, order=np.flip(bins), orient='h', ax=ax, color=f_color)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel('Age group')
                        ax.set_yticklabels(labels[1:])
                        xticks = ax.get_xticks()
                        if percentages:
                            xlabels = [f'{abs(i):.2f}' for i in xticks]
                        else:
                            xlabels = [f'{sc.sigfig(abs(i), sigfigs=2, SI=True)}' for i in xticks]
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xlabels)
                        ax.set_title(f'{date} - data')

                    count += 1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)



class age_results(Analyzer):
    '''
    Constructs results by age at specified points within the sim. Can be used with data
    Args:
        timepoints  (list): list of ints/strings/date objects, timepoints at which to generate by-age results
        results     (list): list of strings, results to generate
        age_standardized (bool): whether or not to provide age-standardized results
        compute_fit (bool): whether or not to compute fit between model results and data
        die         (bool): whether or not to raise an exception if errors are found
        kwargs      (dict): passed to Analyzer()

    **Example**::
        sim = hp.Sim(analyzers=hp.age_results(timepoints=['2015', '2020'], results=['hpv_incidence', 'total_cancers']))
        sim.run()
        age_results = sim['analyzers'][0]
    '''

    def __init__(self, timepoints, edges=None, result_keys=None, age_labels=None, age_standardized=False, datafile=None,
                 compute_fit=False, die=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timepoints          = sc.promotetolist(timepoints) # Combine multiple timepoints
        self.timepoints     = timepoints
        self.edges          = edges # Edges of bins
        self.datafile       = datafile # Data file to load
        self.bins           = None # Age bins, calculated from edges
        self.data           = None # Store the loaded data
        self.die            = die  # Whether or not to raise an exception
        self.dates          = None # Representations in terms of years, e.g. 2020.4, set during initialization
        self.start          = None # Store the start year of the simulation
        self.age_labels     = age_labels # Labels for the age bins - will be automatically generated if not provided
        self.age_standard   = None
        self.age_standardized = age_standardized # Whether or not to compute age-standardized results
        self.compute_fit    = compute_fit # Whether or not to compute fit
        self.result_keys    = result_keys # Store the result keys
        self.results        = sc.odict() # Store the age results
        return


    def initialize(self, sim):

        super().initialize()

        # Handle timepoints and dates
        self.start = sim['start']   # Store the simulation start
        self.end   = sim['end']     # Store simulation end
        if self.timepoints is None:
            self.timepoints = self.end # If no day is supplied, use the last day
        self.timepoints, self.dates = sim.get_t(self.timepoints, return_date_format='str') # Ensure timepoints and dates are in the right format
        max_hist_time = self.timepoints[-1]
        max_sim_time = sim['end']
        if max_hist_time > max_sim_time:
            errormsg = f'Cannot create age results for {self.dates[-1]} ({max_hist_time}) because the simulation ends on {self.end} ({max_sim_time})'
            raise ValueError(errormsg)

        # Handle dt - if we're storing annual results we'll need to aggregate them over
        # several consecutive timesteps
        self.dt = sim['dt']
        self.calcpoints = []
        for tp in self.timepoints:
            self.calcpoints += [tp+i for i in range(int(1/self.dt))]

        # Handle edges, age bins, and labels
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins
        if self.age_labels is None:
            self.age_labels = [f'{int(self.bins[i])}-{int(self.bins[i+1])}' for i in range(len(self.bins)-1)]
            self.age_labels.append(f'{int(self.bins[-1])}+')

        if self.age_standardized:
            self.age_standard = self.get_standard_population()
            self.edges = self.age_standard[0]
            self.bins = self.edges[:-1]  # Don't include the last edge in the bins
            self.age_labels = [f'{int(self.bins[i])}-{int(self.bins[i + 1])}' for i in range(len(self.bins) - 1)]
            self.age_labels.append(f'{int(self.bins[-1])}+')
        else:
            self.age_standard = np.array([self.edges, np.full(len(self.edges), 1)])

        # Handle result keys
        choices = sim.result_keys()
        if self.result_keys is None:
            self.result_keys = ['total_cancers'] # Defaults
        for rk in self.result_keys:
            if rk not in choices:
                strm = '\n'.join(choices)
                errormsg = f'Cannot compute age results for {rk}. Please enter one of the standard sim result_keys to the age_results analyzer; choices are {strm}.'
                raise ValueError(errormsg)

        # Store genotypes
        self.ng = sim['n_genotypes']
        self.glabels = [g.label for g in sim['genotypes']]

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = hpm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

            # Validate the data. Currently we only allow the same timepoints and age brackets
            data_dates = {str(float(i)) for i in self.data.year}
            if len(set(self.dates)-data_dates) or len(data_dates-set(self.dates)):
                string = f'Dates provided in the age result datafile ({data_dates}) are not the same as the age result dates that were requested ({self.dates}).'
                if self.die:
                    raise ValueError(string)
                else:
                    string += '\nPlots will only show requested dates, not all dates in the datafile.'
                    print(string)
            self.data_dates = data_dates

            # Validate the edges - must be the same as requested edges from the model output
            data_bins = np.array(self.data.age.unique(), dtype=float)
            if not np.array_equal(np.sort(self.bins), np.sort(data_bins)):
                errormsg = f'Age bins provided in the age result datafile ({data_bins}) are not the same as the age result age bins that were requested ({self.edges}).'
                raise ValueError(errormsg)

            # Validate the result keys - must be the same as requested edges from the model output
            data_result_keys_array = self.data.name.unique()
            data_result_keys = []
            for drk in data_result_keys_array:
                if drk not in self.result_keys:
                    string = f'The age result datafile contains {drk}, which is not one of the results were requested ({self.result_keys}).'
                    if self.die:
                        raise ValueError(string)
                    else:
                        string += '\nPlots will only show requested results, not all results in the datafile.'
                        print(string)
                else:
                    data_result_keys.append(drk)
            self.data_result_keys = data_result_keys


        if self.compute_fit:
            if self.data is None:
                errormsg = f'Cannot compute fit without data'
                raise ValueError(errormsg)
            else:
                if 'weights' in self.data.columns:
                    self.weights = self.data['weights'].values
                else:
                    self.weights = np.ones(len(self.data))
                self.mismatch = None  # The final value

        # Handle variable names (TODO, should this be centralized somewhere?)
        self.mapping = {
            'infections': ['date_infectious', 'infectious'],
            'cin':  ['date_cin1', 'cin'], # Not a typo - the date the get a CIN is the same as the date they get a CIN1
            'cin1': ['date_cin1', 'cin1'],
            'cin2': ['date_cin2', 'cin2'],
            'cin3': ['date_cin3', 'cin3'],
            'cancers': ['date_cancerous', 'cancerous'],
            'cancer': ['date_cancerous', 'cancerous'],
        }

        # Store colors
        self.result_properties = sc.objdict()
        for rkey in self.result_keys:
            self.result_properties[rkey] = sc.objdict()
            self.result_properties[rkey].color = sim.results[rkey].color
            self.result_properties[rkey].name = sim.results[rkey].name

        self.initialized = True

        return


    def get_standard_population(self):
        '''
        Returns the WHO standard population for computation of age-standardized rates
        https://seer.cancer.gov/stdpopulations/world.who.html
        '''

        age_standard = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                                [0.08860, 0.08690, 0.086, 0.0847, 0.0822, 0.0793, 0.0761, 0.0715, 0.0659, 0.0604,
                                 0.0537, 0.0455, 0.0372, 0.0296, 0.0221, 0.0152, 0.0091, 0.0044, 0.0015, 0.0004,
                                 0.00005]])
        return age_standard


    def apply(self, sim):
        ''' Calculate age results '''

        # Shorten variables that are used a lot
        na = len(self.bins)
        ng = self.ng

        # This is the a timepoint where age results were requested. Start by initializing storage
        # If the requested result is a stock, we can just calculate it for this timepoint
        if sim.t in self.timepoints:
            ind = sc.findinds(self.timepoints, sim.t)[0] # Get the index
            date = self.dates[ind] # Create the date which will be used to key the results
            self.results[date] = sc.objdict() # Initialize the dictionary for result storage
            self.results[date]['bins'] = self.bins # Copy here for convenience
            age = sim.people.age # Get the age distribution
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor


            for rkey in self.result_keys: # Loop over each result, but only stocks are calculated here

                if self.compute_fit and 'total' not in rkey:
                    thisdatadf = self.data[(self.data.year == float(date)) & (self.data.name == rkey)]
                    unique_genotypes = thisdatadf.genotype.unique()
                    ng = len(unique_genotypes)
                # Initialize storage
                size = na if 'total' in rkey else (ng,na)
                self.results[date][rkey] = np.zeros(size)

                # Both annual stocks and prevalence require us to calculate the current stocks.
                # Unlike incidence, these don't have to be aggregated over multiple timepoints.
                if rkey[0] == 'n' or 'prevalence' in rkey:
                    attr = rkey.replace('total_','').replace('_prevalence','') # Name of the actual state
                    if attr[0] == 'n': attr = attr[2:]
                    if attr == 'hpv': attr = 'infectious' # People with HPV are referred to as infectious in the sim
                    if attr == 'cancer': attr = 'cancerous'
                    if attr in sim.people.keys():
                        if 'total' in rkey:
                            inds = sim.people[attr].any(axis=0).nonzero()  # Pull out people for which this state is true
                            self.results[date][rkey] = np.histogram(age[inds[-1]], bins=self.edges)[0] * scale  # Bin the people
                        else:
                            for g in range(ng):
                                inds = sim.people[attr][g,:].nonzero()
                                self.results[date][rkey][g,:] = np.histogram(age[inds[-1]], bins=self.edges)[0] * scale  # Bin the people

                        if 'prevalence' in rkey:
                            # Need to divide by the right denominator
                            if 'hpv' in rkey: # Denominator is whole population
                                denom = (np.histogram(age, bins=self.edges)[0] * scale)
                            else: # Denominator is females
                                denom = (np.histogram(age[sim.people.f_inds], bins=self.edges)[0] * scale)
                            if 'total' not in rkey: denom = denom[None,:]
                            self.results[date][rkey] = self.results[date][rkey] / denom
                    else:
                        if 'detectable' in rkey:
                            hpv_test_pars = hppar.get_screen_pars('hpv')
                            for state in ['hpv', 'cin1', 'cin2', 'cin3', 'cancerous']:
                                for g in range(ng):
                                    hpv_pos_probs = np.zeros(len(sim.people))
                                    tp_inds = hpu.true(sim.people[state][g, :])
                                    hpv_pos_probs[tp_inds] = hpv_test_pars['test_positivity'][state][
                                        sim['genotype_map'][g]]
                                    hpv_pos_inds = hpu.true(hpu.binomial_arr(hpv_pos_probs))
                                    self.results[date][rkey][g, :] += np.histogram(age[hpv_pos_inds], bins=self.edges)[0] * scale  # Bin the people
                            denom = (np.histogram(age, bins=self.edges)[0] * scale)
                            self.results[date][rkey] = self.results[date][rkey] / denom
            self.date = date # Need to store the date for subsequent calcpoints

        # Both annual new cases and incidence require us to calculate the new cases over all
        # the timepoints that belong to the requested year.
        if sim.t in self.calcpoints:
            date = self.date # Stored just above for use here
            scale = sim.rescale_vec[sim.t//sim.resfreq] # Determine current scale factor
            age = sim.people.age # Get the age distribution
            age_standard = self.age_standard[1, :-1]

            for rkey in self.result_keys: # Loop over each result

                if self.compute_fit and 'total' not in rkey:
                    thisdatadf = self.data[(self.data.year == float(date)) & (self.data.name == rkey)]
                    unique_genotypes = thisdatadf.genotype.unique()
                    ng = len(unique_genotypes)

                # Figure out if it's a flow or incidence
                if rkey.replace('total_', '') in hpd.flow_keys or 'incidence' in rkey:
                    attr = rkey.replace('total_','').replace('_incidence','') # Name of the actual state
                    if attr == 'hpv': attr = 'infections' # HPV is referred to as infections in the sim
                    if attr == 'cancer': attr = 'cancers' # cancer is referred to as cancers in the sim
                    attr1 = self.mapping[attr][0] # Messy way of turning 'total cancers' into 'date_cancerous' and 'cancerous' etc
                    attr2 = self.mapping[attr][1] # As above
                    if rkey[:5] == 'total': # Results across all genotypes
                        inds = ((sim.people[attr1]==sim.t)*(sim.people[attr2])).nonzero()
                        self.results[date][rkey] += np.histogram(age[inds[-1]], bins=self.edges)[0] * scale * age_standard  # Bin the people
                    else: # Results by genotype
                        for g in range(ng): # Loop over genotypes
                            inds = ((sim.people[attr1][g,:] == sim.t) * (sim.people[attr2][g,:])).nonzero()
                            self.results[date][rkey][g,:] += np.histogram(age[inds[-1]], bins=self.edges)[0] * scale * age_standard  # Bin the people

                    if 'incidence' in rkey:
                        # Need to divide by the right denominator
                        if 'hpv' in rkey: # Denominator is susceptible population
                            denom = (np.histogram(age[sim.people.sus_pool[-1]], bins=self.edges)[0] * scale)
                        else:  # Denominator is females
                            denom = (np.histogram(age[sim.people.f_inds], bins=self.edges)[0] * scale * 1/100000)  # scale to be per 100,000 for cancer and cin incidence
                        if 'total' not in rkey: denom = denom[None,:]
                        self.results[date][rkey] = self.results[date][rkey] / denom


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.results.keys(), die=self.die)
        if self.compute_fit:
            self.mismatch = self.compute()
            sim.fit = self.mismatch
        return


    def compute(self):
        res = []
        for name, group in self.data.groupby(['name', 'genotype', 'year']):
            key = name[0]
            genotype = name[1].lower()
            year = str(name[2]) + '.0'
            if 'total' in key:
                sim_res = list(self.results[year][key])
                res.extend(sim_res)
            else:
                sim_res = list(self.results[year][key][self.glabels.index(genotype)])
                res.extend(sim_res)
        self.data['model_output'] = res
        self.data['diffs'] = self.data['model_output'] - self.data['value']
        self.data['gofs'] = hpm.compute_gof(self.data['value'].values, self.data['model_output'].values)
        self.data['losses'] = self.data['gofs'].values * self.weights
        self.mismatch = self.data['losses'].sum()

        return self.mismatch


    def plot(self, fig_args=None, axis_args=None, data_args=None, width=0.8,
             do_save=None, fig_path=None, do_show=True, **kwargs):
        '''
        Plot the age results

        Args:
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            width (float): width of the bars
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hp.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        all_args = sc.mergedicts(fig_args, axis_args, d_args)

        # Initialize
        fig = pl.figure(**fig_args)

        # Handle what to plot
        if not len(self.results):
            errormsg = f'Cannot plot since no age results were recorded (scheduled timepoints: {self.timepoints})'
            raise ValueError(errormsg)
        if len(self.timepoints)>1:
            n_cols = len(self.timepoints) # One column for each requested timepoint
            n_rows = len(self.result_keys) # One row for each requested result
        else: # If there's only one timepoint, automatically figure out rows and columns
            n_plots = len(self.result_keys)
            n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Make the figure(s)
        with hpo.with_style(**kwargs):
            col_count=1
            for date,resdict in self.results.items():

                pl.subplots_adjust(**axis_args)
                row_count = 0

                for rkey in self.result_keys:

                    # Start making plot
                    barwidth = 1 * width
                    ax = pl.subplot(n_rows, n_cols, col_count+row_count)
                    x = np.arange(len(self.age_labels))  # the label locations
                    if self.data is not None:
                        thisdatadf = self.data[(self.data.year == float(date))&(self.data.name == rkey)]
                        unique_genotypes = thisdatadf.genotype.unique()
                        # if len(thisdatadf)>0:
                        #     barwidth /= 2 # Adjust width based on data

                    if 'total' not in rkey:
                        # Prepare plot settings
                        barwidth /= self.ng  # Adjust width based on number of genotypes (warning, this will be crowded)
                        if (self.ng % 2) == 0:  # Incredibly complex way of automatically generating bar offsets
                            xlocations = np.array([-(g + 1) for g in reversed(range(self.ng // 2))] + [(g + 1) for g in range(self.ng // 2)]) * .5 * barwidth
                        else:
                            xlocations = np.array([-2 * (g + 1) for g in reversed(range(self.ng // 2))] + [0] + [2 * (g + 1) for g in range(self.ng // 2)]) * .5 * barwidth

                        for g in range(self.ng):
                            glabel = self.glabels[g].upper()
                            ax.plot(x, resdict[rkey][g,:], color=self.result_properties[rkey].color[g], linestyle='--', label=f'Model - {glabel}')
                            # ax.bar(x+xlocations[g], resdict[rkey][g,:], color=self.result_properties[rkey].color[g], label=f'Model - {glabel}', width=barwidth)
                            if len(thisdatadf)>0:
                                # check if this genotype is in dataframe
                                if self.glabels[g].upper() in unique_genotypes:
                                    ydata = np.array(thisdatadf[thisdatadf.genotype==self.glabels[g].upper()].value)
                                    ax.scatter(x, ydata, color=self.result_properties[rkey].color[g], marker='s', label=f'Data - {glabel}')

                    else:
                        if (self.data is not None) and (len(thisdatadf) > 0):
                            ax.plot(x, resdict[rkey], color=self.result_properties[rkey].color, linestyle='--', label='Model')
                            # ax.bar(x-1/2*barwidth, resdict[rkey], color=self.result_properties[rkey].color, width=barwidth, label='Model')
                            ydata = np.array(thisdatadf.value)
                            ax.scatter(x, ydata,  color=self.result_properties[rkey].color, marker='s', label='Data')
                        else:
                            # ax.bar(x, resdict[rkey], color=self.result_properties[rkey].color, width=barwidth, label='Model')
                            ax.plot(x, resdict[rkey], color=self.result_properties[rkey].color, linestyle='--', label='Model')
                    ax.set_xlabel('Age group')
                    ax.set_title(self.result_properties[rkey].name+' - '+date)
                    ax.legend()
                    row_count += n_cols
                    ax.set_xticks(x, self.age_labels)

                col_count+=1


        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)


def import_optuna():
    ''' A helper function to import Optuna, which is an optional dependency '''
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op


class Calibration(Analyzer):
    '''
    A class to handle calibration of HPVsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org), which must be installed separately (via
    pip install optuna).

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        sim          (Sim)  : the simulation to calibrate
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        par_samplers (dict) : an optional mapping from parameters to the Optuna sampler to use for choosing new points for each; by default, suggest_uniform
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers)
        name         (str)  : the name of the database (default: 'hpvsim_calibration')
        db_name      (str)  : the name of the database file (default: 'hpvsim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        rand_seed    (int)  : if provided, use this random seed to initialize Optuna runs (for reproducibility)
        label        (str)  : a label for this calibration object
        die          (bool) : whether to stop if an exception is encountered (default: false)
        verbose      (bool) : whether to print details of the calibration
        kwargs       (dict) : passed to hpv.Calibration()

    Returns:
        A Calibration object

    **Example**::

        sim = hpv.Sim(datafile='data.csv')
        calib_pars = dict(beta=[0.015, 0.010, 0.020])
        calib = hpv.Calibration(sim, calib_pars, total_trials=100)
        calib.calibrate()
        calib.plot()

    '''

    def __init__(self, sim, calib_pars=None, fit_args=None, par_samplers=None,
                 n_trials=None, n_workers=None, total_trials=None, name=None, db_name=None,
                 keep_db=None, storage=None, rand_seed=None, label=None, die=False, verbose=True):
        super().__init__(label=label) # Initialize the Analyzer object

        import multiprocessing as mp # Import here since it's also slow

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name      is None: name      = 'hpvsim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = total_trials/n_workers
        self.run_args   = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name, keep_db=keep_db, storage=storage, rand_seed=rand_seed)

        # Handle other inputs
        self.sim          = sim
        self.calib_pars   = calib_pars
        self.fit_args     = sc.mergedicts(fit_args)
        self.par_samplers = sc.mergedicts(par_samplers)
        self.die          = die
        self.verbose      = verbose
        self.calibrated   = False
        self.results = []

        # Create age_results intervention
        data = self.sim.data
        timepoints = data.year.unique()
        keys = data.name.unique()
        edges = np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.])
        ar = age_results(timepoints=timepoints, result_keys=keys, edges=edges,
                         datafile='test_data/south_africa_target_data.xlsx',
                         compute_fit=True)
        self.sim['analyzers'] += [ar]

        return

    def run_sim(self, calib_pars, label=None, return_sim=False):
        ''' Create and run a simulation '''
        sim = self.sim.copy()
        if label: sim.label = label
        valid_pars = {k:v for k,v in calib_pars.items() if k in sim.pars}
        if 'prognoses' in valid_pars.keys():
            sim_progs = hppar.get_prognoses()
            for prog_key, prog_val in valid_pars['prognoses'].items():
                sim_progs[prog_key] = prog_val
            valid_pars['prognoses'] = sim_progs
        sim.update_pars(valid_pars)
        if len(valid_pars) != len(calib_pars):
            extra = set(calib_pars.keys()) - set(valid_pars.keys())
            errormsg = f'The following parameters are not part of the sim, nor is a custom function specified to use them: {sc.strjoin(extra)}'
            raise ValueError(errormsg)
        try:
            sim.run()
            sim.compute_fit()
            a = sim.get_analyzer()
            self.results.append(a.results)
            if return_sim:
                return sim
            else:
                return sim.fit

        except Exception as E:
            if self.die:
                raise E
            else:
                warnmsg = f'Encountered error running sim!\nParameters:\n{valid_pars}\nTraceback:\n{sc.traceback()}'
                hpm.warn(warnmsg)
                output = None if return_sim else np.inf
                return output

    def run_trial(self, trial):
        ''' Define the objective for Optuna '''
        pars = {}
        for key, val in self.calib_pars.items():
            if isinstance(val, list):
                low, high = val[1], val[2]
                if key in self.par_samplers:  # If a custom sampler is used, get it now
                    try:
                        sampler_fn = getattr(trial, self.par_samplers[key])
                    except Exception as E:
                        errormsg = 'The requested sampler function is not found: ensure it is a valid attribute of an Optuna Trial object'
                        raise AttributeError(errormsg) from E
                else:
                    sampler_fn = trial.suggest_uniform
                pars[key] = sampler_fn(key, low, high)  # Sample from values within this range
            elif isinstance(val, dict):
                sampler_fn = trial.suggest_uniform
                pars[key] = dict()
                for parkey, par_highlowlist in val.items():
                    pars[key][parkey] = []
                    for i, (best, low, high) in enumerate(par_highlowlist):
                        sampler_key = parkey+str(i)
                        sample = float(sampler_fn(sampler_key, low, high))
                        pars[key][parkey].append(sample)

        mismatch = self.run_sim(pars)
        # a = sim.get_analyzer()
        # self.results.append(a.results)
        return mismatch


    def worker(self):
        ''' Run a single worker '''
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials)
        return output


    def run_workers(self):
        ''' Run multiple workers in parallel '''
        if self.run_args.n_workers > 1: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output


    def remove_db(self):
        '''
        Remove the database file if keep_db is false and the path exists.

        New in version 3.1.0.
        '''
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            if self.verbose:
                print(f'Removed existing calibration {self.run_args.db_name}')
        return


    def make_study(self):
        ''' Make a study, deleting one if it already exists '''
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        if self.run_args.rand_seed is not None:
            sampler = op.samplers.RandomSampler(self.run_args.rand_seed)
            sampler.reseed_rng()
            raise NotImplementedError('Implemented but does not work')
        else:
            sampler = None
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler)
        return output


    def calibrate(self, calib_pars=None, verbose=True, **kwargs):
        '''
        Actually perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        '''
        op = import_optuna()

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        if self.calib_pars is None:
            errormsg = 'You must supply calibration parameters either when creating the calibration object or when calling calibrate().'
            raise ValueError(errormsg)
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        self.study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        self.best_pars = sc.objdict(self.study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        # Compare the results
        self.initial_pars = sc.objdict({k:v[0] for k,v in self.calib_pars.items()})
        self.par_bounds   = sc.objdict({k:np.array([v[1], v[2]]) for k,v in self.calib_pars.items()})
        self.before = self.run_sim(calib_pars=self.initial_pars, label='Before calibration', return_sim=True)
        self.after  = self.run_sim(calib_pars=self.best_pars,    label='After calibration', return_sim=True)
        self.parse_study()

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()
        if verbose:
            self.summarize()

        return self


    def summarize(self):
        ''' Print out results from the calibration '''
        if self.calibrated:
            print(f'Calibration for {self.run_args.n_workers*self.run_args.n_trials} total trials completed in {self.elapsed:0.1f} s.')
            before = self.before.fit
            after = self.after.fit
            print('\nInitial parameter values:')
            print(self.initial_pars)
            print('\nBest parameter values:')
            print(self.best_pars)
            print(f'\nMismatch before calibration: {before:n}')
            print(f'Mismatch after calibration:  {after:n}')
            print(f'Percent improvement:         {((before-after)/before)*100:0.1f}%')
            return before, after
        else:
            print('Calibration not yet run; please run calib.calibrate()')
            return


    def parse_study(self):
        '''Parse the study into a data frame -- called automatically '''
        best = self.best_pars

        print('Making results structure...')
        results = []
        n_trials = len(self.study.trials)
        failed_trials = []
        for trial in self.study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    hpm.warn(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)

        return


    def to_json(self, filename=None):
        '''
        Convert the data to JSON.

        New in version 3.1.1.
        '''
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        if filename:
            sc.savejson(filename, json, indent=2)
        else:
            return json