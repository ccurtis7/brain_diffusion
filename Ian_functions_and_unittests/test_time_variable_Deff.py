"""Unit tests for time_variable_Deff.py"""

import os
import pandas as pd
import sys
import unittest

from StringIO import StringIO

import time_variable_Deff


class TestDataCleaning(unittest.TestCase):


    def test_duplicate_removal(self):
        # Part of cleaning these data (and any data, since the
        # subsequent functions in this file require no duplicate
        # columns) is to create a list of particle chemistries with the
        # duplicates (originally there for biological replicates)
        # removed, yielding one column title for each chemistry.  This
        # test verifies that there are no duplicates in this list.
        vc = pd.Series(time_variable_Deff.columns2).value_counts()
        self.assertEqual(vc[vc > 1].index.tolist(), [])

    def test_data_size(self):
        # There is a possibility that test_image_file_output won't be
        # adequate (if the data table is blank I believe
        # make_scatter_plot() will still save a blank figure), so I
        # want to also test if the data variable is as big as we expect.
        trips = pu.get_trip_data()
        rows = len(trips)
        self.assertEqual(rows, 142846)


if __name__ == '__main__':
    unittest.main()
