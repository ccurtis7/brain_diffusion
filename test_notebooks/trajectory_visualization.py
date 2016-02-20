from bokeh.io import output_notebook
from bokeh.plotting import figure, show, gridplot
import numpy as np


def download_trajectory_data(file):
    """
    Downloads a specified datafile and converts to a numpy array.  If the file
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
            return 'File does not exist'
            file_exists = False
        else:
            return name

    else:

        file_csv = False
        return 'File is not a .csv file'


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

    times = dataset[:, 0]
    Run = dict()

    for num in range(1, sets + 1):
        Run["Run" + str(num)] = dataset[:, 3 * num - 2: 3 * num + 1]

    return times, Run


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

        return "Array contains data that isn't of type float64."

    else:

        # Checks if array is correct format (3 columns)
        if width == 3:

            x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
            y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

            xydata[:, 1] = xydata[:, 1] - x_mean
            xydata[:, 2] = xydata[:, 2] - y_mean

            return "Trajectory successfully shifted."

        else:

            if width > 3:

                x_mean = (np.min(xydata[:, 1]) + np.max(xydata[:, 1]))/2
                y_mean = (np.min(xydata[:, 2]) + np.max(xydata[:, 2]))/2

                xydata[:, 1] = xydata[:, 1] - x_mean
                xydata[:, 2] = xydata[:, 2] - y_mean

                justright = False

                return "Array has more than three columns. May not yield correct results"

            else:

                justright = False
                return "Array doesn't have enough columns"


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
        p = figure(title=charttitle, title_text_font_size='8pt',
                   width=300, height=300, x_axis_label='x', y_axis_label='y')
        p.line(x, y, line_width=2)
        show(p)

    else:

        justright = False

        if width > 3:

            x = xydata[:, 1]
            y = xydata[:, 2]
            p = figure(title=charttitle, title_text_font_size='8pt',
                       width=300, height=300, x_axis_label='x', y_axis_label='y')
            p.line(x, y, line_width=2)
            show(p)

            return "Array has more than three columns.  May not yield correct results"

        else:

            justright = False
            return "Array doesn't have enough columns"
