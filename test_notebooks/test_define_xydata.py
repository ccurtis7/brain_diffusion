''' Unit tests for prime.py '''

import unittest
import wget
import urllib
from pronto_utils import download_if_needed2, get_pronto_data, plot_daily_totals, get_trip_data, get_weather_data, trips_by_date, get_trips_and_weather
from pronto_utils import remove_data

class TestDownloadifNeeded(unittest.TestCase):

  # Test whether the file was downloaded
  def testPlotSuccessful(self):
    # 1 website
    result = plot_daily_totals()
    self.assertTrue(result[0])


if __name__ == '__main__':
    unittest.main()
