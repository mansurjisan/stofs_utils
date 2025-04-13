# tests/test_json_utils.py

import json
import os
import unittest
import tempfile
import pandas as pd
from datetime import datetime
from stofs_utils.io import json_utils

class TestJSONUtils(unittest.TestCase):
    
    def setUp(self):
        self.test_data = {
            "stations": [
                {"station_id": "001", "name": "Test Station 1", "lon": -70.0, "lat": 40.0, "type": "gauge"},
                {"station_id": "002", "name": "Test Station 2", "lon": -71.0, "lat": 41.0, "type": "buoy"}
            ],
            "metadata": {
                "created_at": datetime(2024, 4, 11).isoformat()
            }
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "test.json")
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_save_and_load_json(self):
        json_utils.save_json(self.test_data, self.test_file)
        loaded_data = json_utils.load_json(self.test_file)
        self.assertEqual(loaded_data["stations"][0]["name"], "Test Station 1")
    
    def test_json_to_dataframe_and_back(self):
        df = pd.DataFrame(self.test_data["stations"])
        json_back = json_utils.dataframe_to_json(df)
        self.assertEqual(json_back[0]["name"], "Test Station 1")
    
    def test_create_station_json(self):
        df = pd.DataFrame(self.test_data["stations"])
        output_file = os.path.join(self.temp_dir.name, "stations.json")
        stations_dict = json_utils.create_station_json(df, output_file)
        self.assertIn("001", stations_dict)
        self.assertTrue(os.path.exists(output_file))

if __name__ == "__main__":
    unittest.main()
