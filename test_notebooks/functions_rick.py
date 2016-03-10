"""
This file extracts the effective diffusion (Deff) and particle property (PP) data from the
zipfile and assigns them to variables. Size and zeta potential are then
categorized into designated increments, adding 2 columns to the particle
property data. The Deff and PP data are then indexed by NP type and joined
into one data set.
"""

import numpy as np
import pandas as pd
import zipfile
from bokeh.charts import Histogram, output_notebook, show, defaults

# Unzip the csv data and assign it to variables
zf = zipfile.ZipFile('brain-diffusion_data.zip')
file_handle1 = zf.open('Effective_Diffusion_1s_Data.csv')
deff = pd.read_csv(file_handle1)
deff = deff.set_index('Particle')
deff = deff.transpose()
file_handle2 = zf.open('Particle_Properties_Data.csv')
prop = pd.read_csv(file_handle2)

def set_size_range(center,tick):
    """
    Function with input of "central" and "tick" as float  to create categories of
    particle size.
    "center" designates the central measure of the NP in nanometers.
    "tick" designates the range in nanometers between two ticks of the x-axis.
    For example, "center: 100, tick: 10" sets 100 as the central measurement, and
    creates category divides at 90, 80, 70...110, 120, 130...etc.
    Function returns "sizes": a new dataframe with the low and high size limits of
    each tick range, and a third column of categories.
    For example, 'Low': 70, 'High': 80, 'Size_Range': '70 to 80'
    """
    low = center
    while low > min(prop['Size']):
        low = low-tick
    high = center
    while high < max(prop['Size']):
        high = high+tick
    rows = (high-low)/tick
    rows = int(rows)
    # The extra row and column of sizes_array are used as indices when the
    # array is converted to DataFrame
    sizes_array = np.zeros((rows+1,4))
    sizes = pd.DataFrame(sizes_array[1:,1:])
    sizes.columns = ['Low','High','Size_Range']
    for x in range(0,rows):
        sizes.loc[[x],['Low']] = low+tick*x
        sizes.loc[[x],['High']] = low+tick*(x+1)
        sizes.loc[[x],['Size_Range']] = '%i to %i' % (sizes['Low'][x], sizes['High'][x])
    return sizes

def set_zp_range(tick):
    """
    Function with input of tick range to create categories of zeta potential from
    lowest value to zero. These range from some negative value to zero.
    "tick" designates the range in zeta potential (zp) between two ticks of the x-axis.
    For example, "tick: 2" creates category divides at 0, -2, -4, -6...etc.
    Function returns "zp": a new dataframe with the low and high zp limits of
    each tick range, and a third column of categories.
    For example, 'Low': -4, 'High': -2, 'ZP_Range': '-4 to -2'
    """
    low = 0
    while low > min(prop['Zeta_Potential']):
        low = low-tick
    rows = (0-low)/tick
    rows = int(rows)
    zp_array = np.zeros((rows+1,4))
    zp = pd.DataFrame(zp_array[1:,1:])
    zp.columns = ['Low','High','ZP_Range']
    for x in range(0,rows):
        zp.loc[[x],['Low']] = low+tick*x
        zp.loc[[x],['High']] = low+tick*(x+1)
        zp.loc[[x],['ZP_Range']] = '%i to %i' % (zp['Low'][x], zp['High'][x])
    return zp

# function assigning size and zeta potential categories to each particle
# property row that then merges Deff and PP to create a working data set
def prop_data(size_center,size_tick,zp_tick):
    prop['Size_Range'] = 0
    prop['ZP_Range'] = 0
    prop['Color'] = 0
    size_range = set_size_range(size_center,size_tick)
    zp_range = set_zp_range(zp_tick)
    for x in range(0, len(prop)):
        for y in range(0, len(size_range)):
            if prop['Size'][x] >= size_range['Low'][y] and prop['Size'][x] < size_range['High'][y]:
                prop.loc[[x],['Size_Range']] = size_range['Size_Range'][y]
                break
        for z in range(0,len(zp_range)):
            if prop['Zeta_Potential'][x] >= zp_range['Low'][z] and prop['Zeta_Potential'][x] < zp_range['High'][z]:
                prop.loc[[x],['ZP_Range']] = zp_range['ZP_Range'][z]
                break
    """
    deffnew = set_bin_counts()
    deff2 = deffnew.set_index('Particle')
    prop2 = prop.set_index('Sample')
    data = prop2.join(deff2)
    return data
    """
    prop2 = prop.set_index('Sample')
    return prop2

def working_data(size_center,size_tick,zp_tick):
    zf = zipfile.ZipFile('brain-diffusion_data.zip')
    file_handle1 = zf.open('Effective_Diffusion_1s_Data.csv')
    deff = pd.read_csv(file_handle1)
    deff = deff.set_index('Particle')
    deff = deff.transpose()
    p1 = pd.melt(deff, value_vars=['PLGA58k UP'])
    p2 = pd.melt(deff, value_vars=['PLGA58k P80'])
    p3 = pd.melt(deff, value_vars=['PLGA58k F68'])
    p4 = pd.melt(deff, value_vars=['PLGA58k 5CHA'])
    p5 = pd.melt(deff, value_vars=['PLGA15k UP'])
    p6 = pd.melt(deff, value_vars=['PLGA15k P80'])
    p7 = pd.melt(deff, value_vars=['PLGA15k F68'])
    p8 = pd.melt(deff, value_vars=['PLGA15k 5CHA'])
    p9 = pd.melt(deff, value_vars=['PLGA15k 2CHA'])
    p10 = pd.melt(deff, value_vars=['PLGA15k 0.5CHA'])
    p11 = pd.melt(deff, value_vars=['PEG-PLGA45k UP'])
    p12 = pd.melt(deff, value_vars=['PEG-PLGA45k P80'])
    p13 = pd.melt(deff, value_vars=['PEG-PLGA45k F68'])
    p14 = pd.melt(deff, value_vars=['PEG-PLGA45k 5CHA'])
    p15 = pd.melt(deff, value_vars=['PEG-PLGA15k UP'])
    p16 = pd.melt(deff, value_vars=['PEG-PLGA15k P80'])
    p17 = pd.melt(deff, value_vars=['PEG-PLGA15k F68'])
    p18 = pd.melt(deff, value_vars=['PEG-PLGA15k 5CHA'])
    p19 = pd.melt(deff, value_vars=['PEG-PLGA15k 2CHA'])
    p20 = pd.melt(deff, value_vars=['PEG-PLGA15k 0.5CHA'])
    p_all = p1.append([p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
    properties = prop_data(size_center,size_tick,zp_tick)
    properties = properties.reset_index()
    headers = ['Particle_Type','Surfactant','PEG','Size_Range','ZP_Range']
    master = list()
    for i in range(0,5):
        for j in range(0,len(properties)):
            holder = list([properties[headers[i]][j]]*484)
            master = master + holder
        masterseries = pd.Series(master)
        p_all[headers[i]]= pd.Series(master, index=p_all.index)
        master=list()
        holder=list()
    p_all.rename(columns={'value': 'Deff'}, inplace=True)
    p_all = p_all[p_all.Deff > 0]
    return p_all

def plot_Deff(size_center,size_tick,zp_tick,bins_num):
    data = working_data(size_center,size_tick,zp_tick)
    defaults.width = 1000
    p = Histogram(data, values='Deff', color='Particle', bins=bins_num, title="Deff Distribution of Particles", legend='top_right')
    output_notebook()
    show(p)

"""
def set_deff_bin():
    deff['MaxDeff'] = deff[:].max(axis=1)
    high = max(deff['MaxDeff'])
    bin_size = high/10
    deff_bin_array = np.zeros((11,4))
    deff_bin = pd.DataFrame(deff_bin_array[1:,1:])
    deff_bin.columns = ['Low','High','Deff_Range']
    for x in range(0,10):
        deff_bin.loc[[x],['Low']] = bin_size*x
        deff_bin.loc[[x],['High']] = bin_size*(x+1)
        deff_bin.loc[[x],['Deff_Range']] = '%f to %f' % (deff_bin['Low'][x], deff_bin['High'][x])
    return deff_bin

def set_bin_counts():
    deffnew = deff.ix[:,0:11]
    deff_bin = set_deff_bin()
    n=0
    total=0
    for x in range(0, 20):
        for y in range(1,11):
            deffnew.loc[[x],['%i' % (y)]] = 0
    deff_bin = set_deff_bin()
    for x in range(0, len(deffnew)):
        for y in range(0,len(deff_bin)):
            n=0
            total=0
            for z in range(1,len(deff.columns)-1):
                if deff['%i' % (z)][x] >= deff_bin['Low'][y] and deff['%i' % (z)][x] < deff_bin['High'][y]:
                    n=n+1
                if deff['%i' % (z)][x]>=0:
                    total=total+1
            deffnew.loc[[x],['%i' % (y+1)]] = n/total
    for x in range(0,len(deff_bin)):
        deffnew=deffnew.rename(columns = {'%i' % (x+1):deff_bin['Deff_Range'][x]})
    return deffnew
"""
