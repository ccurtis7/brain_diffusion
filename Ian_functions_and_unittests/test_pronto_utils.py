"""Unit tests for pronto_utils.py"""

import unittest
import pronto_utils as pu
import os
import sys
from StringIO import StringIO


class TestDownloadIfNeeded(unittest.TestCase):

    def test_when_file_exists(self):
        # Temporarily writing the system outputs to a variable to be called in the assertion
        backup = sys.stdout
        result = StringIO()
        sys.stdout = result
        # Now, call the function to be tested, using a file guaranteed to exist.
        # I could also have done this by calling pu.get_pronto_data() and then
        # using the open_data_year_one.zip file as my argument in download_if_needed(),
        # but I thought this would be easier and wouldn't be limited by download times.
        pu.download_if_needed('https://s3.amazonaws.com/pronto-data/open_data_year_one.zip','pronto_utils.py')
        sys.stdout = backup
        expected_print = 'pronto_utils.py already exists\n'
        # Check if it printed the right string given that the file doesn't exist
        self.assertEqual(result.getvalue(), expected_print)

    def test_when_file_doesnt_exist(self):
        # First we need to guarantee that the file does not exist
        if os.path.exists('open_data_year_one.zip'):
            pu.remove_data()
        # Write the output to a variable as above
        backup = sys.stdout
        result = StringIO()
        sys.stdout = result
        pu.download_if_needed('https://s3.amazonaws.com/pronto-data/open_data_year_one.zip','open_data_year_one.zip')
        sys.stdout = backup
        expected_print = 'Downloading open_data_year_one.zip ...\n...success\n'
        self.assertEqual(result.getvalue(), expected_print)


class TestRemoveData(unittest.TestCase):

    def test_when_file_exists(self):
        # Make sure file does actually exist
        pu.get_pronto_data()
        pu.remove_data()
        # Assign a result that reflects the existence of the file
        # Note: below, in test_image_file_output, I realized I made this way longer
        # than it had to be.  I left this noob version intact, for posterity.
        if os.path.exists('open_data_year_one.zip'):
            result = True
        else:
            result = False
        self.assertFalse(result)

    # def test_when_file_doesnt_exist(self):
    # I was about to write out this function, and then I realized its testing would rely
    # on its functionality, so this test function is rather pointless.  If I could tell the
    # remove_data() function a filename as an argument, I could just plug in 'nonsense.zip'
    # and test the printed string, but the function isn't built with an argument.  Oh well.


class TestPlotDailyTotals(unittest.TestCase):

    def test_image_file_output(self):
        # This test will check if plot_daily_totals() actually got to the point of
        # saving the figure to the directory, assuming that everything must have
        # functioned correctly up to that point.
        # First, have to make sure the image file is freshly created.
        if os.path.exists('daily_totals.png'):
            os.unlink('daily_totals.png')
        pu.plot_daily_totals()
        self.assertTrue(os.path.exists('daily_totals.png'))
        # Woohoo!  Here's the short version of that assert statement from above.

    def test_data_size(self):
        # There is a possibility that test_image_file_output won't be adequate (if
        # the data table is blank I believe plot_daily_totals() will still save a
        # blank figure), so I want to also test if the data variable is as big as we expect.
        data = pu.get_trips_and_weather()
        rows = len(data)
        self.assertEqual(rows, 365)


if __name__ == '__main__':
    unittest.main()
