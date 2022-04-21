''' Import crude birth rates from WB '''

import wbgapi as wb
import sciris as sc
import numpy as np

birth_rates = wb.data.DataFrame('SP.DYN.CBRT.IN', time=range(2015,2020), labels=True, skipAggs=True).reset_index()
d = dict()
for country in birth_rates['Country'].unique():
    d[country] = birth_rates.loc[(birth_rates['Country']==country)].values[0,2:]
d['years'] = np.arange(2015,2020)
sc.saveobj('data/birth_rates.obj',d)

    