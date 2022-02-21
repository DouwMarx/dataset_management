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
        self.assertIsInstance(result, np.ndarray)


class TestPhenomenologicalDataBase(unittest.TestCase):

    def test_build_phenomenological_database(self):
        from dataset_management.phenomenological_model.add_raw_data_to_database import main
        result_len, n_severities = main()
        self.assertEqual(result_len, n_severities)


class TestProcessed(unittest.TestCase):

    def test_build_processed(self):
        from dataset_management.general.update_database_with_processed import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)


class TestAugment(unittest.TestCase):

    def test_build_augmented(self):
        from dataset_management.general.update_data_base_with_augmented import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)

class TestEncode(unittest.TestCase):

    def test_build_encoding(self):
        from dataset_management.general.update_database_with_encodings import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)


class TestFeaturesCompute(unittest.TestCase):

    def test_update_database_with_computed_features(self):
        from database_definitions import processed
        docs = processed.find_one({})  # Check if there are any documents that are not time series

        self.assertTrue(docs is not None)
