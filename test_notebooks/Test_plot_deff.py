''' Unit tests for plot_deff '''

import unittest
import os
import pandas as pd
import zipfile
from functions_rick import obtain_data
from functions_rick import set_size_range
from functions_rick import set_zp_range
from functions_rick import prop_data
from functions_rick import working_data
from functions_rick import plot_deff
from IPython.display import clear_output, display, HTML
from bokeh.charts import Histogram, output_notebook, show, defaults

class Test_plot_deff(unittest.TestCase):

  def testAllParticlesIncluded(self):
    data = plot_deff(10,'All','All','All','All','All')

    self.assertTrue(data == 20)

if __name__ == '__main__':
    unittest.main()
