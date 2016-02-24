"""
A script for calculating Deff based on MSD data from t=0 to a user-selected
timepoint or range of timepoints (in which case Deff is a geometric mean).
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from bokeh.plotting import figure, output_file, show
import zipfile


ALPHA = 1


zf = zipfile.ZipFile('brain-diffusion_data.zip')
file_handle = zf.open('Mean_Square_Displacement_Data.csv')
msd = pd.read_csv(file_handle)
# Rename columns without biological replicate number (this allows for
# averaging of biological replicates later)
msd.columns = msd.iloc[0]
msd = msd.iloc[1:]
# Create a list of column names without replicates
columns = msd.columns
columns = columns[1:67]
columns2 = [ii for n, ii in enumerate(columns) if ii not in columns[:n]]


"""
The following function combines MSDs of three biological replicates
(different brain tissue samples) using a geometric mean.
"""
def compute_geomean(df, ColumnName):
    geomeans = [0]
    # Grab the three MSD values, one timepoint at a time
    for i in range(2, len(df)+1):
        timepoint = df[ColumnName].ix[i]
        bioreps = list(timepoint[[0,1,2]])
        # Convert strings to floats
        biorepsfloat = []
        for j in bioreps:
            floatMSD = float(j)
            biorepsfloat.append(floatMSD)
        # Append to a growing list of geometric means, then add as
        # a new column in the original dataframe
        geomeans.append(scipy.stats.gmean(biorepsfloat))
    df[ColumnName+' geo'] = geomeans


# Create a column of mean MSDs for each particle type
for title in columns2:
    compute_geomean(msd, title)
# Reset the index to timepoints, converting from string to float
msd = msd.set_index('Particle')
tempindex = [0.0]
for i in range(1, len(msd)):
    tempindex.append(float(msd.index[i]))
msd['index'] = tempindex
msd = msd.set_index('index')


"""
The following function trims the dataframe to the user-selected
timepoint range, calculates Deff from mean MSD, plots a histogram of
the list of Deffs, and gives a geometric mean Deff value for the time
range specified.
"""
def compute_hist_Deff(ParticleChemistry,tmin,tmax):
    # Trim out-of-time-range rows
    temp1_msd = msd[msd.index >= tmin]
    temp2_msd = temp1_msd[temp1_msd.index <= tmax]
    # Calculate Deffs for only the timepoints needed and add as a new
    # column
    Deff_list = []
    for i in range(0, len(temp2_msd)):
        index = temp2_msd.index[i]
        Deff_list.append(temp2_msd[ParticleChemistry + ' geo'][index]/(4*index**ALPHA))
    temp2_msd['Deff'] = Deff_list
    # Plot histogram and print mean Deff value
    # NOTE: Eventually I'll migrate the plot to bokeh; I'm using
    # matplotlib temporarily for ease of testing
    plt.hist(temp2_msd['Deff'], bins=15)
    plt.xlabel('Calculated Deffs')
    plt.ylabel('Count')
    plt.show()
    Deff = scipy.stats.gmean(temp2_msd['Deff'])
    return Deff


def compute_plot_all_Deff(tmin,tmax):
    # Trim out-of-time-range rows
    temp1_msd = msd[msd.index >= tmin]
    temp2_msd = temp1_msd[temp1_msd.index <= tmax]
    # Calculate Deffs for only the timepoints needed and add as a new
    # column to a new dataframe
    index = temp2_msd.index
    Deffs = pd.DataFrame(index=index, columns=columns2)
    avg_Deffs = pd.DataFrame(index=columns2)
    avg_Deffs_temp = []
    output_file('Deffs_plot.html')
    p = figure(tools='resize,pan,box_zoom,wheel_zoom,reset,save', x_axis_label='MSD timepoint', y_axis_label='Deff')
    for title in columns2:
        single_Deff_list = []
        for i in range(0, len(temp2_msd)):
            index = temp2_msd.index[i]
            single_Deff_list.append(temp2_msd[title + ' geo'][index]/(4*index**ALPHA))
        Deffs[title] = single_Deff_list
        avg_Deffs_temp.append(scipy.stats.gmean(Deffs[title]))
        p.line(Deffs.index, Deffs[title], legend=title)
    avg_Deffs['Deff'] = avg_Deffs_temp
    p.legend.label_text_font_size = '6pt'
    # p.legend.label_width = 50
    # p.legend.label_height = 6
    show(p)
    print Deffs
    return avg_Deffs
