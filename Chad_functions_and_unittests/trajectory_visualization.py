from bokeh.io import output_notebook
from bokeh.plotting import figure, show, gridplot, hplot, vplot, curdoc
import numpy as np
import os
import csv
from bokeh.client import push_session
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def download_trajectory_data(file):
    """
    Converts a .csv file dataset to a numpy array.  If the file
    is not a .csv file or the file does not exist, returns an error message.

    Inputs: .csv file.
    Outputs: numpy array
    """

    # Define unit test variables
    file_csv = True
    file_exists = True

    # Check if file is a .csv file
    if file[-4:] == '.csv':

        # Check if file exists and downloads it.\
        try:
            name = np.genfromtxt(file, delimiter=",")
            name = np.delete(name, 0, 0)
        except:
            print('File does not exist')
            file_exists = False
            return (file_csv, file_exists)
        else:
            return name

    else:

        file_csv = False
        print('File is not a .csv file')
        return (file_csv, file_exists)


def csv_writer():
    """
    Creates a sample dataset for unit tests called output.csv
    """
    if os.path.exists('output.csv'):
        print('output.csv', 'already exists')

    else:

        data = ["first_name,last_name,city".split(","),
                "Tyrese,Hirthe,Strackeport".split(","),
                "Jules,Dicki,Lake Nickolasville".split(","),
                "Dedric,Medhurst,Stiedemannberg".split(",")
                ]
        path = "output.csv"
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)


def define_xydata(dataset, sets):
    """
    Takes a large numpy array (presumably trajectory data with frames, x
    coordinates, y coordinates) of n columns and splits into (n-1)/3 numpy
    arrays of x-y data plus a separate array of time data.

    The first column in array is time column.  Afterwards, every set of three
    columns becomes an entry in dictionary (Run i).

    Example: trajectory data over 10-second interval with three particles, each
    with three types of data: frame, x coordinate, y coordinate.  define_xydata
    would output a times array (1 column) plus 3 3-column arrays of trajectory
    data with names Run['Run1'], Run['Run2'], Run['Run3'].  These can be used
    to define variable with desired names later.

    Input: Large numpy array (n columns), number of sets to define from array
    Output: time array (1 column)
            trajectory arrays ((n-1)/3 columns)
    """

    splitsuccess = True
    wholenumber = True
    justrightsets = True
    correctformat = True

    if (dataset.shape[1] - 1) % 3 == 0:

        if sets < (dataset.shape[1] - 1)/3:

            if float(sets).is_integer():

                times = dataset[:, 0]
                Run = dict()

                for num in range(1, sets + 1):
                    Run["Run" + str(num)] = dataset[:, 3 * num - 2: 3 * num + 1]

                return times, Run

            else:
                print("The variable 'sets' must be a whole number.")
                wholenumber = False
                splitsuccess = False
                return (splitsuccess, wholenumber, justrightsets, correctformat)

        else:
            print("Desired number of sets is too high for this dataset.")
            justrightsets = False
            splitsuccess = False
            return (splitsuccess, wholenumber, justrightsets, correctformat)

    else:
        print("Data is not of right format.  Dataset must have 1 time column and sets of 3 columns of trajectory data.")
        correctformat = False
        splitsuccess = False
        return (splitsuccess, wholenumber, justrightsets, correctformat)


def shift_trajectory(xydata):
    """
    Adjusts the coordinate system of x-y trajectory data such that if the data
    were plotted, the center of the plot would be (0,0).

    Inputs:
    xydata: a numpy array of 3 columns: frame, x-coordinate and y-coordinate.
    """

    length = xydata.shape[0]
    width = xydata.shape[1]

    # Define unit test variables
    numerical = True
    justright = True

    # checks for data that's not float64 format
    for num in range(0, width):

        for yes in range(0, length):

            if np.dtype(xydata[yes, num]) == np.dtype('float64'):
                numerical = True
            else:
                numerical = False
                break

    if not numerical:

        print("Array contains data that isn't of type float64.")

    else:

        # Checks if array is correct format (3 columns)
        if width == 3:

            x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
            y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

            xydata[:, 1] = xydata[:, 1] - x_mean
            xydata[:, 2] = xydata[:, 2] - y_mean

            print("Trajectory successfully shifted.")

        else:

            if width > 3:

                x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
                y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

                xydata[:, 1] = xydata[:, 1] - x_mean
                xydata[:, 2] = xydata[:, 2] - y_mean

                justright = False

                print("Array has more than three columns. May not yield correct results")

            else:

                justright = False
                print("Array doesn't have enough columns")

    return (numerical, justright)


def shift_trajectory3D(xyzdata):
    """
    Adjusts the coordinate system of x-y-z trajectory data such that if the data
    were plotted, the center of the plot would be (0,0,0).

    Inputs:
    xyzdata: a numpy array of 3 columns: frame, x-coordinate and y-coordinate.
    """

    length = xyzdata.shape[0]
    width = xyzdata.shape[1]

    # Define unit test variables
    numerical = True
    justright = True

    # checks for data that's not float64 format
    for num in range(0, width):

        for yes in range(0, length):

            if np.dtype(xyzdata[yes, num]) == np.dtype('float64'):
                numerical = True
            else:
                numerical = False
                break

    if not numerical:

        print("Array contains data that isn't of type float64.")

    else:

        # Checks if array is correct format (3 columns)
        if width == 3:

            x_mean = (np.min(xyzdata[:, 0]) + np.max(xyzdata[:, 0]))/2
            y_mean = (np.min(xyzdata[:, 1]) + np.max(xyzdata[:, 1]))/2
            z_mean = (np.min(xyzdata[:, 2]) + np.max(xyzdata[:, 2]))/2

            xyzdata[:, 0] = xyzdata[:, 0] - x_mean
            xyzdata[:, 1] = xyzdata[:, 1] - y_mean
            xyzdata[:, 2] = xyzdata[:, 2] - z_mean

            print("Trajectory successfully shifted.")

        else:

            if width > 3:

                x_mean = (np.min(xyzdata[:, 0]) + np.max(xyzdata[:, 0]))/2
                y_mean = (np.min(xyzdata[:, 1]) + np.max(xyzdata[:, 1]))/2
                z_mean = (np.min(xyzdata[:, 2]) + np.max(xyzdata[:, 2]))/2

                xydata[:, 1] = xyzdata[:, 1] - x_mean
                xydata[:, 2] = xyzdata[:, 2] - y_mean

                justright = False

                print("Array has more than three columns. May not yield correct results")

            else:

                justright = False
                print("Array doesn't have enough columns")

    return (numerical, justright)


def plot_trajectory(xydata, charttitle):
    """
    Plots a single 3-column numpy array of trajectory data (frames in column 1,
    x coordinates in column 2, y coordinates in column 3) within iPython
    notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: numpy array, chart title (string)
    Output: displays trajectory plot inline
    """

    length = xydata.shape[0]
    width = xydata.shape[1]

    justright = True

    if width == 3:

        x = xydata[:, 1]
        y = xydata[:, 2]
        p = figure(title=charttitle, title_text_font_size='13pt',
                   width=300, height=300)
        p.line(x, y, line_width=2)
        show(p)

    else:

        justright = False

        if width > 3:

            x = xydata[:, 1]
            y = xydata[:, 2]
            p = figure(title=charttitle, title_text_font_size='13pt',
                       width=300, height=300)
            p.line(x, y, line_width=2)
            show(p)

            print("Array has more than three columns.  May not yield correct results")

        else:

            justright = False
            print("Array doesn't have enough columns")

    return justright


def sidebyside(xydata1, xydata2, charttitle1, charttitle2):
    """
    Plots two 3-column numpy arrays of trajectory data (frames in column 1, x
    coordinates in column 2, y coordinates in column 3) next to each other in
    iPython notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: 2 numpy arrays, 2 chart titles (strings)
    Output: displays trajectory plots inline.
    """
    length1 = xydata1.shape[0]
    width1 = xydata1.shape[1]

    justright = True

    length2 = xydata2.shape[0]
    width2 = xydata2.shape[1]

    if width1 == 3 and width2 == 3:

        x1 = xydata1[:, 1]
        y1 = xydata1[:, 2]
        x2 = xydata2[:, 1]
        y2 = xydata2[:, 2]

        s1 = figure(title=charttitle1, title_text_font_size='13pt', width=300, height=300)
        s1.line(x1, y1, color='navy', line_width=2)

        s2 = figure(title=charttitle2, title_text_font_size='13pt', width=300, height=300, x_range=s1.x_range, y_range=s1.y_range)
        s2.line(x2, y2, color='firebrick', line_width=2)

        p = gridplot([[s1, s2]])
        show(p)

    else:

        justright = False

        if width1 > 3 or width2 > 3:

            x1 = xydata1[:, 1]
            y1 = xydata1[:, 2]
            x2 = xydata2[:, 1]
            y2 = xydata2[:, 2]

            s1 = figure(title=charttitle1, width=300, height=300, x_axis_label='x', y_axis_label='y')
            s1.line(x1, y1, color='navy', line_width=2)

            s2 = figure(title=charttitle2, width=300, height=300, x_range=s1.x_range, y_range=s1.y_range, x_axis_label='x', y_axis_label='y')
            s2.line(x2, y2, color='firebrick', line_width=2)

            p = hplot(s1, s2)
            show(p)

            print("At least one of the given arrays has more than three columns.  May not yield corect results.")

        else:

            print("One of the given arrays has less than three columns.  Data could not be plotted.")

    return justright


def overlay(xydata1, xydata2, charttitle):
    """
    Plots two 3-column numpy arrays of trajectory data (frames in column 1, x
    coordinates in column 2, y coordinates in column 3) superimposed upon each
    other in iPython notebook.

    Note: MUST have run output_notebook in order to run successfully.

    Input: 2 numpy arrays, 2 chart titles (strings)
    Output: displays trajectory plots inline.
    """

    length1 = xydata1.shape[0]
    width1 = xydata1.shape[1]

    justright = True

    length2 = xydata2.shape[0]
    width2 = xydata2.shape[1]

    if width1 == 3 and width2 == 3:

        x1 = xydata1[:, 1]
        y1 = xydata1[:, 2]
        x2 = xydata2[:, 1]
        y2 = xydata2[:, 2]

        p = figure(title=charttitle, title_text_font_size='9pt', width=300, height=300, x_axis_label='x', y_axis_label='y')

        p.line(x1, y1, line_width=2, color='navy')
        p.line(x2, y2, line_width=2, color='firebrick')

        show(p)

    else:

        justright = False

        if width1 > 3 or width2 > 3:

            x1 = xydata1[:, 1]
            y1 = xydata1[:, 2]
            x2 = xydata2[:, 1]
            y2 = xydata2[:, 2]

            p = figure(title=charttitle, width=300, height=300)

            p.line(x1, y1, line_width=2, color='navy')
            p.line(x2, y2, line_width=2, color='firebrick')

            show(p)

            print("At least one of the given arrays has more than three columns.  May not yield corect results.")

        else:

            print("One of the given arrays has less than three columns.  Data could not be plotted.")

    return justright


def animated_plot(xydata):
    """
    I haven't been able to generate a working code for an animated plot function
    and I haven't been able to find out why.  Whenever I try, I get an error
    message saying that index cannot be defined.
    """
    b = xydata
    xlist = b[:, 1]
    ylist = b[:, 2]

    # create a plot and style its properties
    p = figure(x_range=(min(xlist), max(xlist)), y_range=(min(ylist), max(ylist)), toolbar_location=None)

    # add a text renderer to out plot (no data yet)
    r = p.line(x=[], y=[], line_width=3, color='navy')

    session = push_session(curdoc())

    index = 0
    ds = r.data_source

    # create a callback that will add the next point of the trajectory data
    def callback():
        global index
        ds.data['x'].append(xlist[index])
        ds.data['y'].append(ylist[index])
        ds.trigger('data', ds.data, ds.data)
        index = index + 1

    curdoc().add_periodic_callback(callback, 67)

    # open the document in a browser
    session.show()

    # run forever
    session.loop_until_closed()


def plot_trajectories3D(traj, n1, n2, n3, dec, filename):
    """
    This function creates a multiple 3D plots from trajectory data.  This
    dataset must include a column of particle numbers as well as the x, y, and
    z coordinates of of each particle at each frame. Output will be saved as a
    .png file of the desired name.
    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: x data
    n3: z data (with y data, of course, in between x and z.  This just defines a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n3+1]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    # Determines arrangement of subplots
    rows = np.sqrt(total)
    if (rows % 1 == 0):
        rows = int(rows)
    else:
        rows = int(rows) + 1

    # Create figure
    fig = plt.figure(figsize=(24, 20), dpi=80)
    ax = dict()

    # Plot trajectories
    for num in range(1, total1):

        number = 100*rows + 10*rows + num
        ax[num] = fig.add_subplot(number, projection='3d')
        ax[num].set_title('Particle {}'.format(num), x=0.5, y=1.08)
        ax[num].plot(path[num][:, 0], path[num][:, 1], path[num][:, 2])
        ax[num].locator_params(nbins=5)
        ax[num].title.set_fontsize(30)

        ax[num].tick_params(direction='out', pad=13)

        ax[num].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
        ax[num].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
        ax[num].zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    # Save figures
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_3Doverlay(traj, n1, n2, dec, filename):
    """
    This function creates a single 3D plot from trajectory data.  This dataset
    must include a column of particle numbers as well as the x, y, and z
    coordinates of of each particle at each frame. Output will be saved as a
    .png file of the desired name.
    Inputs:
    traj: array of trajectory data e.g. particle #, frames, x, y, z, Deff, MSD
    n1: particle# column
    n2: xyz data start (so x data column, 29 for a normal dataset)
    a range)
    dec: how many decimals you would like to be displayed in the graph.
    filename: what you want to name the file.  Must be in ''.
    Can also use plt.show() afterwards to preview your data, even if it skews the title and legend a bit.
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    position = traj[:, n2:n2+4]
    total = int(max(particles))
    total1 = total + 1
    path = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        path[num] = (position[min1:max1, :])

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(path[num][:, 0], path[num][:, 1], path[num][:, 2], label='Particle {}'.format(num))

    axbox = ax.get_position()
    ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    ax.locator_params(nbins=4)
    ax.view_init(elev=38, azim=72)

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(13)

    ax.title.set_fontsize(35)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))
    plt.gca().zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f um'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_MSDorDeff(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs (34 or 35 for my datasets)
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    rawMSD = traj[:, n3]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (rawMSD[min1:max1])
        time[num] = (rawtime[min1:max1])

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    # Plots individual trajectories
    for num in range(1, total1):

        ax.plot(time[num][:], MSD[num][:], label='Particle {}'.format(num))

    axbox = ax.get_position()
    ax.legend(loc=(0.86, 0.90), prop={'size': 20})
    # ax.locator_params(nbins=4)

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def plot_MeanMSDorDeff(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    rawMSD = traj[:, n3]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (rawMSD[min1:max1])
        time[num] = (rawtime[min1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(time[1][:], MMSD[:])

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')


def randtraj(b, s, f, p):
    """
    Builds a single random trajectory.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)

    Output:
    0 particle number
    1 time or frames
    2 magnitude
    3 angle 1
    4 angle 2
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 20))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vector
        ttraject[num, 2] = base + step*random.random()
        # Create Angle Vectors
        ttraject[num, 3] = 2 * pi * random.random()
        ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]*np.sin(ttraject[num, 4])*np.cos(ttraject[num, 3])
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 2]*np.sin(ttraject[num, 4])*np.sin(ttraject[num, 3])
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 2]*np.cos(ttraject[num, 4])

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                                    (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = np.sqrt((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def randtraj2(b, s, f, p):
    """
    Builds a single random trajectory without using spherical coordinates, as
    randtraj does.

    b: base magnitude of single step
    s: variation in step size
    f: number of frames or steps to Takes
    p: particle number (should be 1 for now)

    Output:
    0 particle number
    1 time or frames
    2 x movement
    3 angle 1 (not used)
    4 angle 2 (note used)
    5 x coordinate
    6 y coordinate
    7 z coordinate
    8 centered x coordinate
    9 centered y coordinate
    10 centered z coordinate
    11 MSD
    12 2D xy MSD
    13 2D xz MSD
    14 2D yz MSD
    15 Diffusion Coefficient (Deff)
    16 2D xy Deff
    17 2D xz Deff
    18 2D yz Deff
    19 y movement
    20 z movement
    """

    base = b
    step = s
    pi = 3.14159265359
    frames = f

    ttraject = np.zeros((frames, 22))

    for num in range(1, frames):

        # Create particle number
        ttraject[num, 0] = p
        ttraject[num-1, 0] = p
        # Create frame
        ttraject[num, 1] = 1 + ttraject[num-1, 1]
        # Create magnitude vectors
        ttraject[num, 2] = base*(random.random()-0.5)
        ttraject[num, 19] = base*(random.random()-0.5)
        ttraject[num, 20] = base*(random.random()-0.5)
        # Create Angle Vectors
        # ttraject[num, 3] = 2 * pi * random.random()
        # ttraject[num, 4] = pi * random.random()
        # Build trajectories
        ttraject[num, 5] = ttraject[num-1, 5] + ttraject[num, 2]
        ttraject[num, 6] = ttraject[num-1, 6] + ttraject[num, 19]
        ttraject[num, 7] = ttraject[num-1, 7] + ttraject[num, 20]

    particle = ttraject[:, 0]
    time = ttraject[:, 1]
    x = ttraject[:, 5]
    y = ttraject[:, 6]
    z = ttraject[:, 7]

    ttraject[:, 8] = x - ((max(x)+min(x))/2)
    cx = ttraject[:, 8]
    ttraject[:, 9] = y - ((max(y)+min(y))/2)
    cy = ttraject[:, 9]
    ttraject[:, 10] = z - ((max(z)+min(z))/2)
    cz = ttraject[:, 10]

    # Calculate MSDs and Deffs
    for num in range(1, frames):

        ttraject[num, 11] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2 +
                                    (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 12] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)
        ttraject[num, 13] = np.sqrt((ttraject[num, 8]-ttraject[0, 8])**2 + (ttraject[num, 10]-ttraject[0, 10])**2)
        ttraject[num, 14] = np.sqrt((ttraject[num, 10]-ttraject[0, 10])**2 + (ttraject[num, 9]-ttraject[0, 9])**2)

        ttraject[num, 15] = ttraject[num, 11]/(6*ttraject[num, 1])
        ttraject[num, 16] = ttraject[num, 12]/(4*ttraject[num, 1])
        ttraject[num, 17] = ttraject[num, 13]/(4*ttraject[num, 1])
        ttraject[num, 18] = ttraject[num, 14]/(4*ttraject[num, 1])

    MSD = ttraject[:, 11]
    MSDxy = ttraject[:, 12]
    MSDxz = ttraject[:, 13]
    MSDyz = ttraject[:, 14]

    Deff = ttraject[:, 15]
    Deffxy = ttraject[:, 16]
    Deffxz = ttraject[:, 17]
    Deffyz = ttraject[:, 18]

    return ttraject


def multrandtraj(b, s, f, p):
    """
    Builds an array of multiple trajectories appended to each other. Number of
    trajectories is determined by p.
    """

    parts = p
    one = randtraj2(b, s, f, 1)
    counter = 1

    while counter < p + 1:
        counter = counter + 1
        one = np.append(one, randtraj2(b, s, f, counter), axis=0)

    return one


def plot_Mean2DMSDsorDeff(traj, n1, n2, n3, dec, datatype, filename):
    """
    Plots the MSDs from a trajectory dataset.

    n1: particle numbers
    n2: time
    n3: MSDs or Deffs
    """

    # Creates an array 'particles' that contains the particle number at each frame.
    particles = traj[:, n1]
    total = int(max(particles))
    total1 = total + 1
    rawtime = traj[:, n2]
    raw2DMSDs = traj[:, n3:n3+4]
    MSD = dict()
    time = dict()

    # Creates an array for each trajectory containing all xyz data
    for num in range(1, total1):

        hold = np.where(particles == num)
        itindex = hold[0]
        min1 = min(itindex)
        max1 = max(itindex)
        MSD[num] = (raw2DMSDs[min1:max1, :])
        time[num] = (rawtime[min1:max1])

    MMSD = MSD[1]
    for num in range(2, total1):
        MMSD = MMSD + MSD[num]
    MMSD = MMSD/total1

    # Creates figure
    fig = plt.figure(figsize=(24, 18), dpi=80)
    ax = fig.add_subplot(111)
    # ax.set_title('Particle Trajectories', x=0.5, y=1.15)

    ax.plot(time[1][:], MMSD[:, 0], label='3D')
    ax.plot(time[1][:], MMSD[:, 1], label='2D xy')
    ax.plot(time[1][:], MMSD[:, 2], label='2D xz')
    ax.plot(time[1][:], MMSD[:, 3], label='2D yz')

    # A few adjustments to prettify the graph
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.title.set_fontsize(35)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(datatype)
    ax.tick_params(direction='out', pad=16)
    ax.legend(loc=(0.86, 0.86), prop={'size': 22})
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.{}f'.format(dec)))

    # Save your figure
    plt.savefig('{}.png'.format(filename), bbox_inches='tight')
