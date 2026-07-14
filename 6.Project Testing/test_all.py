import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add the Project Development Phase directory to sys.path dynamically
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../5. Project Development Phase'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import components from project development phase
from utils.preprocessing import load_and_preprocess_data, prepare_train_test_split
from predict import predict_flood

class TestFloodPredictionSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Sets up paths and references to files.
        """
        cls.test_csv_path = os.path.join(src_path, "dataset/flood.csv")
        cls.model_path = os.path.join(src_path, "model/flood_model.pkl")
        cls.scaler_path = os.path.join(src_path, "model/scaler.pkl")

    def test_dataset_generation_and_loading(self):
        """
        Verifies that data preprocessing handles missing records and shapes correctly.
        """
        self.assertTrue(os.path.exists(self.test_csv_path), "Dataset flood.csv must exist.")
        X, y = load_and_preprocess_data(self.test_csv_path)
        
        # Verify shape
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        
        # Verify target is binary
        unique_targets = set(y.unique())
        self.assertTrue(unique_targets.issubset({0, 1}))

    def test_train_test_split_and_scaling(self):
        """
        Verifies that scaler creates normal zero-mean unit-variance arrays.
        """
        X, y = load_and_preprocess_data(self.test_csv_path)
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)
        
        # Scaling check: mean should be close to 0 and std close to 1
        mean_scaled = np.mean(X_train, axis=0)
        std_scaled = np.std(X_train, axis=0)
        
        for m in mean_scaled:
            self.assertAlmostEqual(m, 0.0, places=1)
        for s in std_scaled:
            self.assertAlmostEqual(s, 1.0, places=1)
            
        self.assertTrue(os.path.exists(self.scaler_path))

    def test_prediction_inference(self):
        """
        Runs mock prediction values to verify outputs of class prediction status and probabilities.
        """
        self.assertTrue(os.path.exists(self.model_path), "Trained model must exist.")
        
        # Generate mock input dict
        mock_input = {
            "Annual Rainfall": 2400.0,
            "Cloud Visibility": 45.0,
            "January Rainfall": 20.0,
            "February Rainfall": 25.0,
            "March Rainfall": 40.0,
            "April Rainfall": 80.0,
            "May Rainfall": 180.0,
            "June Rainfall": 450.0,
            "July Rainfall": 620.0,
            "August Rainfall": 580.0,
            "September Rainfall": 320.0,
            "October Rainfall": 110.0,
            "November Rainfall": 40.0,
            "December Rainfall": 20.0
        }
        
        res = predict_flood(mock_input, model_path=self.model_path, scaler_path=self.scaler_path)
        
        self.assertIn("prediction", res)
        self.assertIn("probability", res)
        self.assertIn("status", res)
        
        self.assertIn(res["prediction"], [0, 1])
        self.assertGreaterEqual(res["probability"], 0.0)
        self.assertLessEqual(res["probability"], 1.0)
        self.assertIn(res["status"], ["Flood Detected", "No Flood"])

if __name__ == '__main__':
    unittest.main()
