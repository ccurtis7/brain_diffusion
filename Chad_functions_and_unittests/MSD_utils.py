import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.ma as ma


def get_data_gels(channels, surface_functionalities, media, concentrations, replicates, path):

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


def return_average(data, frames, to_average='YG_nPEG_in_agarose_1x'):
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


def avg_all(data, frames, avg_sets):
    """
    Averages over all replicates in a dataset.

    data is a dictionary defined in function get_data.
    frames is the number of frames in each experiment.
    avg_sets is a dictionary of strings (keys are 0-N)

    Example:
    avg_all(data, 90, {0: 'RED_PEG_in_agarose_0_1x', 1: 'RED_PEG_in_agarose_1x'})
    """

    all_avg = {}
    for keys in avg_sets:
        all_avg[avg_sets[keys]] = return_average(data, frames, avg_sets[keys])
    return all_avg


def get_data_pups(channels, genotypes, pups, surface_functionalities, slices, regions, replicates, path):

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
    path = "./{genotype}/{pup}/{region}/{channel}/geoM2xy_{sample_name}.csv";
    get_data(["RED", "YG"], ["WT", "KO", "HET"], ["P1", "P2", "P3", "P4"],
    ["PEG", "noPEG"], ["S1", "S2", "S3", "S4"], ["cortex", "hipp", "mid"],
    [1, 2, 3, 4, 5], path)
    """

    data = {}
    avg_over_slices_raw = {}
    avg_over_pups_raw = {}
    names_with_replicates = {}
    counter = 0
    counter2 = 0

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        for slic in slices:
                            test_value = "{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup)
                            avg_over_slices_raw[counter] = test_value
                            test_value2 = "{}_{}_{}_{}".format(channel, genotype, surface_functionality, region)
                            avg_over_pups_raw[counter] = test_value2
                            counter = counter + 1
                            sample_name = "{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality, region, pup, slic)
                            for replicate in replicates:
                                sample_name_long = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality,
                                                                                 region, pup, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long
                                counter2 = counter2 + 1
                            filename = path.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name)
                            data[sample_name] = np.genfromtxt(filename, delimiter=",")

    avg_over_slices = {}
    avg_over_pups = {}

    counter = 0
    for key, value in avg_over_slices_raw.items():
        if value not in avg_over_slices.values():
            avg_over_slices[counter] = value
            counter = counter + 1

    counter = 0
    for key, value in avg_over_pups_raw.items():
        if value not in avg_over_pups.values():
            avg_over_pups[counter] = value
            counter = counter + 1

    return data, avg_over_slices, avg_over_pups, names_with_replicates


def return_SD(data, frames=90, SD_frames=[1, 7, 14, 15], to_stdev='YG_nPEG_in_agarose_1x'):
    """
    Finds standard deviation over replicates within a sample at frames specified by SD_frames.

    Parameters:
    to_average is a string.

    Example:
    return_SD(data, 90, [1, 7, 14, 15], 'RED_PEG_in_agarose_10x')
    """

    to_SD = {}
    counter = 0
    for keys in data:
        if to_stdev in keys:
            to_SD[counter] = keys
            counter = counter + 1

    to_SD_num = np.zeros((frames, counter))
    for i in range(0, counter):
        for j in range(0, frames):
            to_SD_num[j, i] = data[to_SD[i]][j]

    answer = np.std(to_SD_num, axis=1)
    answer_sep = answer[SD_frames[:]]

    return answer_sep


def SD_all(data, frames, SD_frames, avg_sets):
    """
    Finds standard deviations over all replicates in a dataset.

    data is a dictionary defined in function get_data.
    frames is the number of frames in each experiment.
    avg_sets is a dictionary of strings (keys are 0-N)

    Example:
    SD_all(data, 90, [1, 7, 14, 15], {0: 'RED_PEG_in_agarose_0_1x', 1: 'RED_PEG_in_agarose_1x'})
    """

    all_SD = {}
    for keys in avg_sets:
        all_SD[avg_sets[keys]] = return_SD(data, frames, SD_frames, avg_sets[keys])
    return all_SD


def range_and_ticks(range_to_graph, to_frame, manual_decimals=False,
                    manual_decimals_val=2):
    """
    This is a useful function that calculates the best range, tick mark interval
    size, and decimals displayed for a given input dataset.

    Inputs:
    range_to_graph: numpy array of data to be graphed.
    to_frame: integer, limits data to be graphed to the range [0, to_frame].
    manual_decimals: True/False.  Allow the user to manually adjust the number of decimals displayed.
    manual_decimals_val: integer.  Number of decimals the user desires.

    Outputs:
    y_range: upper boundary containing the entire range of range_to_graph[0, to_frame].
    ticks: tick interval.
    decimals: number of decimals to display.

    This function must be modified if I need to plot graphs with negative values.
    I should also include symmetrical capabilities when I want to plot negative values.
    """

    graph_max = np.nanmax(range_to_graph[0:to_frame])  # Define max within range

    if np.ceil(np.log10(graph_max)) >= 0:  # Find nearest 10^x that contains graph_max
        base = np.ceil(np.log10(graph_max))
        raw_max = 10**base
        decimals = int(1)
    else:
        base = np.ceil(np.log10(1/graph_max))
        raw_max = 10**(-base)
        decimals = int(base + 1)

    range_correct = False
    cut_down_max = 0.1*raw_max
    counter = -1
    while range_correct is False:  # Make sure that most of the graph space is used efficiently (75%).
        if graph_max/raw_max > 0.75:
            y_range = raw_max
            range_correct = True
        else:
            raw_max = raw_max - cut_down_max
            range_correct = False
        counter = counter + 1

    if graph_max > y_range:  # Make sure that graph_max doesn't exceed limits
        y_range = y_range + cut_down_max

    if counter % 2 == 0:  # Define tick size. I based it off of 0.1*raw_max, which is always a 10^x
        ticks = cut_down_max * 2
    else:
        ticks = cut_down_max

    range_correct = False  # Modifies the tick size if there aren't enough ticks on the graph.
    while range_correct is False:
        if ticks/y_range >= 0.24:
            ticks = ticks/2
            range_correct = False
        else:
            range_correct = True

    if manual_decimals:  # Allow user to set manual decimal values.
        decimals = manual_decimals_val

    return y_range, ticks, decimals


def choose_y_axis_params(all_avg, in_name1, in_name2, to_frame):
    """
    When plotting multiple trajectories, I need to choose plot parameters based
    on multiple datasets.  This function uses the function ranges_and_ticks to
    choose the best parameters based on multiple trajectories.

    Inputs:
    all_avg: dictionary of numpy arrays.  A subset of this data will be plotted
        based on the parameters in_name1 and in_name2.
    in_name1: string.  in_name1 should be found within the keys in all_avg of
        all datasets to be plotted e.g. if I want to plot all RED datasets, the
        I could use in_name1=RED.
    in_name2: similar criteria to in_name1.  If I want to narrow the data to be
        plotted to PEG datasets, then, in_name2 could be "_PEG" (if you only use
        PEG, it will plot all nPEG and PEG datasets, due to my chosen
        nomenclature. Just be aware of your naming system).
    to_frame: integer, limits data to be graphed to the range [0, to_frame].

    Outputs:
    y_range_final: upper boundary containing the entire range of
        range_to_graph[0, to_frame].
    ticks_final: tick interval.
    decimals_final: number of decimals to display.

    Example:
    choose_y_axis_params(all_avg, "RED", "_PEG", 15)
    """

    to_graph = {}
    counter = 0

    for keys in all_avg:
        if in_name1 in keys and in_name2 in keys:
            to_graph[counter] = keys
            counter = counter + 1

    y_range = np.zeros(counter)
    ticks = np.zeros(counter)
    decimals = np.zeros(counter)
    base = np.zeros(counter)

    for keys in to_graph:
        y_range[keys], ticks[keys], decimals[keys] = range_and_ticks(all_avg[to_graph[keys]], to_frame)

    one_to_graph = np.argmax(y_range)
    y_range_final = y_range[one_to_graph]
    ticks_final = ticks[one_to_graph]
    decimals_final = int(decimals[one_to_graph])

    return y_range_final, ticks_final, decimals_final


def data_prep_for_plotting_pups(path, frames, SD_frames, conversion, to_frame, parameters):
    """
    A summary function that preps my mean mean squared displacement data for graphing by performing averages over slices and
    pups, creates time arrays, calculates standard deviations, and organized nomenclature.

    Inputs:

    path: string. A path to where the MMSD data resides e.g. path = "./{genotype}/geoM2xy_{sample_name}.csv"
    frames: integer.  Number of frames expected in MMSD datasets e.g. frames = 60
    SD_frames: array of integers.  Desired frames at which to plot standard deviation bars e.g. SD_frames = [1, 7, 14, 15]
    conversion: list of three floats.  First is the microns per pixel, second is the frames per second, and third is the microns
        per slice e.g. conversion = (0.3, 3.95, 1)
    to_frame: integer.  Frame to which you which to plot.
    parameters: a dictionary of the form:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["genotypes"] = ["WT"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4, 5]

    Outputs:

    data: dictionary of arrays, with keys being names of individual datasets and entries being the MSDs of the datasets.
    avg_over_slices: numbered dictionary with entries being names of datasets averaged over slices.
    avg_over_pups: numbered dictionary with entries being names of datasets averages over pups.
    names_with_replicates: numbered dictionaries with entries being raw names of datasets (original names without
        averaging over replicates.)
    time: array.  Time array calculated based on frames and frames per second conversion.
    time_SD: array.  Time array with entries at times when standard deviations are calculated.
    average_over_slices: dictionary of arrays containing data averaged over slices.
    average_over_pups: dictionary of arrays containing data averaged over pups.
    all_SD_over_slices: dictionary of arrays containing standard deviations calculated over slices.
    all_SD_over_pups: dictionary of arrays containing standard deviations calculated over pups.

    """

    data = {}
    avg_sets = {}
    all_avg = {}
    all_SD = {}
    names_with_replicates = {}

    data, avg_over_slices, avg_over_pups, names_with_replicates = get_data_pups(parameters["channels"], parameters["genotypes"], parameters["pups"],
                                                                                parameters["surface functionalities"], parameters["slices"],
                                                                                parameters["regions"], parameters["replicates"], path)
    time, time_SD = build_time_array(frames, conversion, SD_frames)
    average_over_slices = avg_all(data, frames, avg_over_slices)
    average_over_pups = avg_all(average_over_slices, frames, avg_over_pups)
    all_SD_over_slices = SD_all(data, frames, SD_frames, avg_over_slices)
    all_SD_over_pups = SD_all(average_over_slices, frames, SD_frames, avg_over_pups)

    return data, avg_over_slices, avg_over_pups, names_with_replicates, time, time_SD,\
        average_over_slices, average_over_pups, all_SD_over_slices, all_SD_over_pups


def graph_single_variable(all_avg, all_SD, time, time_SD, SD_frames, in_name1, in_name2, to_frame=15,
                          line_colors=['g', 'r', 'b', 'c', 'm', 'k'], line_kind='-', x_manual=False,
                          y_range=10, ticks_y=2, dec_y=0, x_range=5, ticks_x=1, dec_x=0, label_size=95,
                          legend_size=40, tick_size=50, line_width=10, fig_size=(20, 18),
                          modify_labels=False, label_identifier="agarose_", base_name="KO"):
    """
    A handy plotting function to examine individual datasets within the larger dataset.

    Inputs:
    all_avg: dictionary of numpy arrays.  A subset of this data will be plotted
        based on the parameters in_name1 and in_name2. Keys contain the name of
        the dataset.
    all_SD: dictionary of numpy arrays.  Mirrors the all_avg dataset.
    time: numpy array.  Assumes time array is the same between datasets
        contained in all_avg.
    time_SD: numpy array.  Contains timepoints at which SDs are desired for the
        plots.
    SD_frames: numpy array.  Contains frames at which SDs are desired for the
        plots.
    in_name1: string.  in_name1 should be found within the keys in all_avg of
        all datasets to be plotted e.g. if I want to plot all RED datasets, the
        I could use in_name1=RED.
    in_name2: similar criteria to in_name1.  If I want to narrow the data to be
        plotted to PEG datasets, then, in_name2 could be "_PEG" (if you only use
        PEG, it will plot all nPEG and PEG datasets, due to my chosen
        nomenclature. Just be aware of your naming system).

    to_frame: integer, limits data to be graphed to the range [0, to_frame].
    line_colors: list of strings.  Contains colors desired for plot.
    line_kind: string.  Contains a single string for the desired line format
        of all lines in graph.
    x_manual: True/False.  If True, the user must define range, ticks, and
        decimals for both x and y.
    y_range: y range of the plot.
    ticks_y: interval size on y axis.
    dec_y: number of decimals displayed on y axis.
    x_range" x range of the plot.
    ticks_x: intervals size on x axis.
    dec_x: number of decimals displayed on the x axis.
    label_size: Size of labels on x and y axes.
    legend_size: size of text in legend.
    tick_size: size of text of ticks.
    line_width: width of lines on plot.
    fig_size: x,y size of plot e.g. (20, 18).
    modify_labels: True/False.  If True, the user must give a label_identifier.
    label_identifier: string.  User puts in a string that is contained in all keys
        of all_avg.  Labels of data will now be whatever follows label_identifier
        e.g. if a key contains "agarose_S1" and I use "agarose_", the legend
        will read "S1".

    To do:
    I need to make my labelling more flexible eventually.  The troubl is correctly
    lining up my labels with the data in all_avg.
    I also need to make it possible to plot in a desired order e.g. 0.1x, 1x, 10x.
    """

    to_graph = {}
    line_type = {}
    counter = 0
    labels = {}

    if not x_manual:
        y_range, ticks_y, dec_y = choose_y_axis_params(all_avg, in_name1, in_name2, to_frame)
        x_range, ticks_x, dec_x = range_and_ticks(time, to_frame)
    else:
        y_range, ticks_y, dec_y = y_range, ticks_y, dec_y
        x_range, ticks_x, dec_x = x_range, ticks_x, dec_x

    filename = base_name + "_" + in_name1 + "_" + in_name2 + ".png"

    # Creates figure
    fig = plt.figure(figsize=fig_size, dpi=80)
    ax = fig.add_subplot(111)

    for keys in all_avg:
        if in_name1 in keys and in_name2 in keys:
            to_graph[counter] = keys
            counter = counter + 1

    for keys in to_graph:
        if modify_labels:
            labels[keys] = to_graph[keys].split(label_identifier, 1)[1]
        else:
            labels[keys] = to_graph[keys]
        line_type[keys] = line_colors[keys]+line_kind
        ax.plot(time[0:to_frame], all_avg[to_graph[keys]][0:to_frame], line_type[keys], linewidth=line_width, label=labels[keys])
        ax.errorbar(time_SD, all_avg[to_graph[keys]][SD_frames], all_SD[to_graph[keys]], fmt='', linestyle='',
                    capsize=7, capthick=2, elinewidth=2, color=line_colors[keys])

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(tick_size)

    xmajor_ticks = np.arange(0, x_range+0.0001, ticks_x)
    ymajor_ticks = np.arange(0, y_range+0.0001, ticks_y)

    ax.set_xticks(xmajor_ticks)
    plt.xticks(rotation=-30)
    ax.set_yticks(ymajor_ticks)
    ax.title.set_fontsize(tick_size)
    ax.set_xlabel('Time (s)', fontsize=label_size)
    ax.set_ylabel(r'MSD ($\mu$m$^2$)', fontsize=label_size)
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.02, 0.75), prop={'size': legend_size})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec_x)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec_y)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([0, x_range+0.0001])
    plt.gca().set_ylim([0, y_range+0.0001])

    # Save your figure
    plt.savefig('{}'.format(filename), bbox_inches='tight')
    return y_range, ticks_y, dec_y, x_range, ticks_x, dec_x


def fill_in_and_split(data):
    """
    This function takes a raw trajectory datafile (with the first column deleted),
    splits it into particles, frames, x data, and y data arrays, and fills in
    any skipped frames.

    Inputs:
    data: array.  First column are particle numbers, second frames, third x data,
        forth, y data.

    Outputs:
    particles_new
    framed_new
    x_data_new
    y_data_new
    """

    particles = data[:, 0]
    framed = data[:, 1].astype(np.int64)
    x_data = data[:, 2]
    y_data = data[:, 3]

    counter = 0
    original_size = framed.size

    for num in range(0, original_size - 1):
        if framed[num+1] > framed[num] and particles[num+1] == particles[num]:
            if not framed[num+1] - framed[num] == 1:
                new_frames = framed[num+1] - framed[num] - 1
                counter = counter + new_frames

    new_size = original_size + counter

    if not counter == 0:
        particles_new = np.zeros(new_size)
        framed_new = np.zeros(new_size)
        x_data_new = np.zeros(new_size)
        y_data_new = np.zeros(new_size)

        counter2 = 0

        for num in range(0, original_size - 1):
            if framed[num+1] > framed[num] and particles[num+1] == particles[num]:
                if not framed[num+1] - framed[num] == 1:
                    new_frames = framed[num+1] - framed[num] - 1
                    for num2 in range(1, new_frames):
                        particles_new[num+num2+counter2] = particles[num]
                        framed_new[num+num2+counter2] = framed[num]
                        x_data_new[num+num2+counter2] = x_data[num]
                        y_data_new[num+num2+counter2] = y_data[num]
                    counter2 = counter2 + new_frames
                else:
                    particles_new[num+counter2] = particles[num]
                    framed_new[num+counter2] = framed[num]
                    x_data_new[num+counter2] = x_data[num]
                    y_data_new[num+counter2] = y_data[num]
            else:
                particles_new[num+counter2] = particles[num]
                framed_new[num+counter2] = framed[num]
                x_data_new[num+counter2] = x_data[num]
                y_data_new[num+counter2] = y_data[num]

        particles_new[original_size+counter2 - 1] = particles[original_size - 1]
        framed_new[original_size+counter2 - 1] = framed[original_size - 1]
        x_data_new[original_size+counter2 - 1] = x_data[original_size - 1]
        y_data_new[original_size+counter2 - 1] = y_data[original_size - 1]
    else:
        particles_new = particles
        framed_new = framed
        x_data_new = x_data
        y_data_new = y_data

    return particles_new, framed_new, x_data_new, y_data_new


def plot_traj_length_histogram(length, total, lenplot):

    total1 = total
    hist = length

    hist = [x for x in hist if str(x) != 'nan']

    plot, bins = np.histogram(hist, bins=total1)
    width = 1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    plt.bar(center, plot, align='center', width=width)
    plt.xlabel('Trajectory lengths', fontsize=20)

    plt.savefig('{}.png'.format(lenplot), bbox_inches='tight')
    return hist, total1


def plot_traj(xts, yts, total, T2plot):
    """
    This function plots the trajectories of the particles as they were originally caught on the
    microscope (no centering performed).

    Inputs:
    xts: dictionary of arrays.  Keys are the particle numbers.
    yts: dictionary of arrays.  Keys are the particle numbers.
    total: total particles in dictionary.
    """
    middle_max_x = np.zeros(total)
    middle_max_y = np.zeros(total)

    for num in range(1, total+1):
        middle_max_x[num-1] = np.nanmax(xts[num])
        middle_max_y[num-1] = np.nanmax(yts[num])

    x_max = np.round(max(middle_max_x)/10, 0)*10+0.1
    y_max = np.round(max(middle_max_y)/10, 0)*10+0.1

    both_max = max(x_max, y_max)

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)

    for num in range(1, total+1):
        ax.plot(xts[num], yts[num], linewidth=3, label='Particle {}'.format(num))

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(70)

    xmajor_ticks = np.arange(0, both_max, 20)
    ymajor_ticks = np.arange(0, both_max, 20)

    ax.set_xticks(xmajor_ticks)
    ax.set_yticks(ymajor_ticks)
    ax.title.set_fontsize(70)
    ax.set_xlabel(r'x ($\mu$m)', fontsize=95)
    ax.set_ylabel(r'y ($\mu$m)', fontsize=95)
    # ax.tick_params(direction='out', pad=16)
    # ax.legend(loc=(0.60, 0.76), prop={'size': 40})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(0)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(0)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([0, both_max])
    plt.gca().set_ylim([0, both_max])

    # Save your figure
    plt.savefig('{}.png'.format(T2plot), bbox_inches='tight')


def filter_out_short_traj(particles_unfiltered, framed_unfiltered, x_data_unfiltered, y_data_unfiltered, cut):
    """
    Filtered out particles trajectories from a dataset that are shorter than the variable cut.

    Inputs:
    particles_unfiltered
    framed_unfiltered
    x_data_unfiltered
    y_data_unfiltered
    cut: integer.

    Outputs:
    particles_filtered
    framed_filtered
    x_data_filtered
    y_data_filtered
    """
    total_unfiltered = int(max(particles_unfiltered))

    counter = 0

    for num in range(1, total_unfiltered+1):

        hold = np.where(particles_unfiltered == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        if max1 - min1 + 1 < cut:
            counter = counter + max1 - min1 + 1

    new_shape = particles_unfiltered.shape[0] - counter
    particles_filtered = np.zeros(new_shape)
    framed_filtered = np.zeros(new_shape)
    x_data_filtered = np.zeros(new_shape)
    y_data_filtered = np.zeros(new_shape)

    counter = 0
    counter2 = 0

    for num in range(1, total_unfiltered+1):

        hold = np.where(particles_unfiltered == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)

        if max1 - min1 + 1 >= cut:
            particles_filtered[min1 - counter:max1+1 - counter] = particles_unfiltered[min1:max1+1] - counter2
            framed_filtered[min1 - counter:max1+1 - counter] = framed_unfiltered[min1:max1+1]
            x_data_filtered[min1 - counter:max1+1 - counter] = x_data_unfiltered[min1:max1+1]
            y_data_filtered[min1 - counter:max1+1 - counter] = y_data_unfiltered[min1:max1+1]
        else:
            counter = counter + max1 - min1 + 1
            counter2 = counter2 + 1

    return particles_filtered, framed_filtered, x_data_filtered, y_data_filtered


def plot_trajectory_overlay(x, y, graph_size, ticks, number_of_trajectories, Tplot):
    """
    This function plots a random selection of trajectories from an x,y trajectory dataset. The user can
    manipulate size of the graph, tick interval size, and the number of trajectories to be plotted.
    """
    unit = graph_size

    maxx = unit + 0.1
    minx = -unit - 0.1
    maxy = unit + 0.1
    miny = -unit - 0.1

    random_particles = np.zeros(number_of_trajectories)
    for num in range(0, number_of_trajectories):
        random_particles[num] = np.random.random_integers(1, max(x))

    dec = 0
    xc = dict()
    yc = dict()
    xcmask = dict()

    # Creates figure
    fig1 = plt.figure(figsize=(24, 18), dpi=80)
    ax1 = fig1.add_subplot(111)

    for num in range(1, random_particles.shape[0]+1):
        lowx = ma.min(x[random_particles[num-1]])
        highx = ma.max(x[random_particles[num-1]])
        lowy = ma.min(y[random_particles[num-1]])
        highy = ma.max(y[random_particles[num-1]])

        xc[random_particles[num-1]] = np.array([x[random_particles[num-1]] - ((highx+lowx)/2)])
        yc[random_particles[num-1]] = np.array([y[random_particles[num-1]] - ((highy+lowy)/2)])

        xcmask[random_particles[num-1]] = x[random_particles[num-1]].recordmask
        xc[random_particles[num-1]] = ma.array(xc[random_particles[num-1]], mask=xcmask[random_particles[num-1]])
        yc[random_particles[num-1]] = ma.array(yc[random_particles[num-1]], mask=xcmask[random_particles[num-1]])

        ax1.plot(xc[random_particles[num-1]][0, :], yc[random_particles[num-1]][0, :],
                 linewidth=10, label='Particle {}'.format(random_particles[num-1]))

    # A few adjustments to prettify the graph
    for item in ([ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(70)

    xmajor_ticks = np.arange(minx, maxx, ticks)
    ymajor_ticks = np.arange(miny, maxy, ticks)

    ax1.set_xticks(xmajor_ticks)
    ax1.set_yticks(ymajor_ticks)
    ax1.title.set_fontsize(70)
    ax1.set_xlabel(r'x ($\mu$m)', fontsize=95)
    ax1.set_ylabel(r'y ($\mu$m)', fontsize=95)
    ax1.tick_params(direction='out', pad=16)
    plt.xticks(rotation=-30)
    # ax1.legend(loc=(0.60, 0.46), prop={'size': 40})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().set_xlim([minx, maxx])
    plt.gca().set_ylim([miny, maxy])

    # Save your figure
    plt.savefig('{}.png'.format(Tplot), bbox_inches='tight')


def quality_control(path2, folder, frames, conversion, parameters, cut):
    """
    This function plots a histogram of trajectory lengths (in units of frames) and two types of plots of
    trajectories (original and overlay).

    Inputs:
    path2: string.  Name of input trajectory csv files.
    folder: string. Name of folder to which to save results.
    frames: integer.  Total number of frames in videos.
    conversion: array.  Contains microns per pixel and frames per second of videos.
    parameters:

    parameters = {}
    parameters["channels"] = ["RED"]
    parameters["genotypes"] = ["WT"]
    parameters["pups"] = ["P1", "P2", "P3"]
    parameters["surface functionalities"] = ["PEG"]
    parameters["slices"] = ["S1", "S2", "S3"]
    parameters["regions"] = ["cortex", "mid"]
    parameters["replicates"] = [1, 2, 3, 4]

    cut: integer.  Minimum number of frames a trajectory must have in order to be plotted.
    """

    channels = parameters["channels"]
    genotypes = parameters["genotypes"]
    pups = parameters["pups"]
    surface_functionalities = parameters["surface functionalities"]
    slices = parameters["slices"]
    regions = parameters["regions"]
    replicates = parameters["replicates"]
    SD_frames = [1, 7, 14, 15]

    trajectory = {}
    names_with_replicates = {}
    data = {}

    particles_unfiltered = {}
    framed_unfiltered = {}
    x_data_unfiltered = {}
    y_data_unfiltered = {}
    total_unfiltered = {}
    particles = {}
    framed = {}
    x_data = {}
    y_data = {}
    total = {}
    tlength = {}
    x_microns = {}
    y_microns = {}
    x_particle = {}
    y_particle = {}

    x_original_frames = {}
    y_original_frames = {}

    x_adjusted_frames = {}
    y_adjusted_frames = {}

    counter2 = 0

    for channel in channels:
        for genotype in genotypes:
            for surface_functionality in surface_functionalities:
                for region in regions:
                    for pup in pups:
                        for slic in slices:
                            for replicate in replicates:

                                # Establishing sample names and extracting data from csv files.
                                counter2 = counter2 + 1
                                sample_name_long = "{}_{}_{}_{}_{}_{}_{}".format(channel, genotype, surface_functionality,
                                                                                 region, pup, slic, replicate)
                                names_with_replicates[counter2] = sample_name_long

                                filename = path2.format(channel=channel, genotype=genotype, pup=pup, region=region, sample_name=sample_name_long)
                                data[sample_name_long] = np.genfromtxt(filename, delimiter=",")
                                data[sample_name_long] = np.delete(data[sample_name_long], 0, 1)

                                # Names of output plots
                                logplot = folder.format(channel=channel, genotype=genotype,
                                                        pup=pup, region=region)+'{}_logplot'.format(sample_name_long)
                                Mplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Mplot'.format(sample_name_long)
                                Dplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Dplot'.format(sample_name_long)
                                Hplot = folder.format(channel=channel, genotype=genotype, pup=pup, region=region)+'{}_Hplot'.format(sample_name_long)
                                Hlogplot = folder.format(channel=channel, genotype=genotype,
                                                         pup=pup, region=region)+'{}_Hlogplot'.format(sample_name_long)
                                Cplot = folder.format(channel=channel, genotype=genotype,
                                                      pup=pup, region=region)+'{}_Cplot'.format(sample_name_long)
                                Tplot = folder.format(channel=channel, genotype=genotype,
                                                      pup=pup, region=region)+'{}_Tplot'.format(sample_name_long)
                                T2plot = folder.format(channel=channel, genotype=genotype,
                                                       pup=pup, region=region)+'{}_T2plot'.format(sample_name_long)
                                lenplot = folder.format(channel=channel, genotype=genotype,
                                                        pup=pup, region=region)+'{}_lenplot'.format(sample_name_long)

                                # Fill in data and split into individual trajectories
                                particles_unfiltered[counter2], framed_unfiltered[counter2], x_data_unfiltered[counter2],\
                                    y_data_unfiltered[counter2] = fill_in_and_split(data[names_with_replicates[counter2]])

                                total_unfiltered[counter2] = int(max(particles_unfiltered[counter2]))

                                # Filter out short trajectories
                                particles[counter2], framed[counter2], x_data[counter2], y_data[counter2] =\
                                    filter_out_short_traj(particles_unfiltered[counter2], framed_unfiltered[counter2],
                                                          x_data_unfiltered[counter2], y_data_unfiltered[counter2], cut)

                                # Convert to microns and seconds
                                time, time_SD = build_time_array(frames, conversion, SD_frames)
                                framen = np.linspace(0, frames, frames+1).astype(np.int64)
                                total[counter2] = int(max(particles[counter2]))
                                tlength[counter2] = np.zeros(total[counter2])

                                x_microns[counter2] = x_data[counter2]*conversion[0]
                                y_microns[counter2] = y_data[counter2]*conversion[0]

                                # Adjust frames (probably unneccesary, but I did it...)
                                x_particle[counter2] = {}
                                y_particle[counter2] = {}

                                x_original_frames[counter2] = {}
                                y_original_frames[counter2] = {}

                                x_adjusted_frames[counter2] = {}
                                y_adjusted_frames[counter2] = {}

                                for num in range(1, total[counter2] + 1):
                                    hold = np.where(particles[counter2] == num)
                                    itindex = hold[0]
                                    min1 = min(itindex)
                                    max1 = max(itindex)
                                    x_particle[counter2][num] = x_microns[counter2][min1:max1+1]
                                    y_particle[counter2][num] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = np.zeros(frames + 1)
                                    y_original_frames[counter2][num] = np.zeros(frames + 1)
                                    x_adjusted_frames[counter2][num] = np.zeros(frames + 1)
                                    y_adjusted_frames[counter2][num] = np.zeros(frames + 1)

                                    x_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = x_microns[counter2][min1:max1+1]
                                    y_original_frames[counter2][num][framed[counter2][min1]:framed[counter2][max1]+1]\
                                        = y_microns[counter2][min1:max1+1]

                                    x_adjusted_frames[counter2][num][0:max1+1-min1] = x_microns[counter2][min1:max1+1]
                                    y_adjusted_frames[counter2][num][0:max1+1-min1] = y_microns[counter2][min1:max1+1]

                                    x_original_frames[counter2][num] = ma.masked_equal(x_original_frames[counter2][num], 0)
                                    y_original_frames[counter2][num] = ma.masked_equal(y_original_frames[counter2][num], 0)
                                    x_adjusted_frames[counter2][num] = ma.masked_equal(x_adjusted_frames[counter2][num], 0)
                                    y_adjusted_frames[counter2][num] = ma.masked_equal(y_adjusted_frames[counter2][num], 0)

                                    tlength[counter2][num - 1] = ma.count(x_adjusted_frames[counter2][num])

                                plot_traj(x_original_frames[counter2], y_original_frames[counter2], total[counter2], T2plot)
                                plt.gcf().clear()
                                plot_trajectory_overlay(x_original_frames[counter2], y_original_frames[counter2], 6, 2, 6, Tplot)
                                plt.gcf().clear()
                                plot_traj_length_histogram(tlength[counter2], max(tlength[counter2]), lenplot)
                                plt.gcf().clear()
