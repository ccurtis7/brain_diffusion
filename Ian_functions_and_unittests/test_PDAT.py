"""Unit tests for pronto_utils.py"""

import os
import sys
import unittest

from StringIO import StringIO

import pdat
import pronto_utils as pu


class TestMakeScatterPlot(unittest.TestCase):


    def test_image_file_output(self):
        # This test will check if make_scatter_plot() actually got to
        # the point of saving the figure to the directory, assuming
        # that everything must have functioned correctly up to that
        # point.  First, have to make sure the image file is freshly
        # created.
        if os.path.exists('durationvsagescatter.pdf'):
            os.unlink('durationvsagescatter.pdf')
        trips = pu.get_trip_data()
        pdat.make_scatter_plot(
        trips,'tripduration','birthyear','durationvsagescatter.pdf')
        self.assertTrue(os.path.exists('durationvsagescatter.pdf'))

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
