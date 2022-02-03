import unittest
from dataset_management.build_data_and_encodings import run_data_and_encoding_pipeline
import numpy as np

class TestBuildDataAndEncodings(unittest.TestCase):

    def test_run_data_and_encoding_pipeline(self):
        run_data_and_encoding_pipeline("data_generated_by_tests", quik_iter=True)


class TestMongoDB(unittest.TestCase):

    def test_store_python_object(self):
        from notebooks.mongo_db_basics import main
        result = main()
        self.assertIsInstance(result,np.ndarray)


