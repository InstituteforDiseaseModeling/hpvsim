'''
Benchmark the simulation

Benchmark results 2022-08-19

initialize(): 20.9%
    init_people(): 97.6%
        make_people(): 98.9%
            get_age_distribution(): 92.4%
                load_file(): 99.8% <- loading a 21 MB file

step(): 78.4%
    update_states_pre(): 15.2%
    create_partnerships(): 24.8%
    get_sources_targets(): 5.2% + 5.2%
    foi_whole: 4.7%
    infect(): 6.7% + 4.6%
    hpv_pos_inds: 9.8%
'''

#%% Imports and settings
import sciris as sc
import hpvsim as hpv
from test_sim import test_sim
from test_interventions import test_new_interventions

sim0 = test_sim(do_plot=False, n_agents=50e3) # For debugging regular sim
sim1 = test_new_interventions() # For debuggin interventions
to_profile = 'apply_int' # Must be one of the options listed below

func_options = {
    'initialize':    sim.initialize,
    'make_contacts': hpv.make_contacts,
    'person':        hpv.Person.__init__,
    'make_people':   hpv.make_people,
    'init_people':   sim.init_people,

    'get_age':       hpv.data.get_age_distribution,
    'run':           sim.run,
    'step':          sim.step,
    'apply_int':     hpv.interventions.treat_num.apply,
    'update_pre':    hpv.people.People.update_states_pre,
    'death':         hpv.people.People.apply_death_rates,
    'infect':        hpv.people.People.infect,
}

sc.profile(run=sim1.run, follow=func_options[to_profile])
