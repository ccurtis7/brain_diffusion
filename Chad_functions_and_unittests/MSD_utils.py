import numpy as np


def get_data(channels, surface_functionalities, media, concentrations, replicates, path):

    """
    Loads data from csv files and outputs a dictionary following a specified
        sample naming convection determined by the input

    Parameters:
    channels, surface functionalities, media, and concentrations, and replicates
        can take ranges or lists.
    path is string with substition placeholders for concentration and sample
        name (built from channels, surface_functionalities, media,
        concentrations, and replicates).

    Example:
    path = "./{concentration}_ACSF/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["PEG", "noPEG"], ["in_agarose"],
    ["0_1x", "1x", "10x"], ["S1", "S2", "S3", "S4"], path)
    """

    data = {}
    avg_sets = {}
    counter = 0

    for channel in channels:
        for surface_functionality in surface_functionalities:
            for medium in media:
                for concentration in concentrations:
                    test_value = "{}_{}_{}_{}".format(channel, surface_functionality, medium, concentration)
                    avg_sets[counter] = test_value
                    counter = counter + 1
                    for replicate in replicates:
                        sample_name = "{}_{}_{}_{}_{}".format(channel, surface_functionality, medium, concentration, replicate)
                        filename = path.format(channel=channel, surface_functionality=surface_functionality, medium=medium,
                                               concentration=concentration, replicate=replicate, sample_name=sample_name)
                        data[sample_name] = np.genfromtxt(filename,
                                                          delimiter=",")

    return data, avg_sets


def build_time_array(frames=90, conversion=(0.16, 9.89, 1), SD_frames=[1, 7, 14]):
    """
    Builds (1) a time array based on the desire number of frames and the fps and (2) a shortened time array at which the
    standard deviations are going to be calculated.

    The default is 100 frames, a conversion factor of 0.16 microns per pixel, 9.89 frames per second, and 1 micron per z-stack.

    Example:
    build_time_array(90, (0.16, 9.89, 1), [1, 7, 14, 15])
    """

    frames_1 = np.linspace(0, frames, frames+1).astype(np.int64)
    time = frames_1/conversion[1]
    time_SD = np.zeros(np.size(SD_frames))

    for i in range(0, np.size(SD_frames)):
        time_SD[i] = time[SD_frames[i]]

    return time, time_SD


def return_average(to_average='YG_nPEG_in_agarose_1x', data):
    """
    Averages over replicates within a sample.

    Parameters:
    to_average is a string.

    Example:
    return_average('RED_PEG_in_agarose_10x')
    """

    to_avg = {}
    counter = 0
    for keys in data:
        if to_average in keys:
            to_avg[counter] = keys
            counter = counter + 1

    to_avg_num = np.zeros((frames, counter))
    for i in range(0, counter):
        for j in range(0, frames):
            to_avg_num[j, i] = data[to_avg[i]][j]

    answer = np.mean(to_avg_num, axis=1)

    return answer
