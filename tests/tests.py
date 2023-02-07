import unittest
import numpy as np

class TestMongoDB(unittest.TestCase):

    def test_store_python_object(self):
        from notebooks.mongo_db_basics import main
        result = main()
        self.assertIsInstance(result, np.ndarray)


class TestPhenomenologicalDataBase(unittest.TestCase):

    def test_build_phenomenological_database(self):
        from dataset_management.phenomenological_model.add_raw_data_to_database import main
        n_docs = main()
        self.assertGreater(n_docs, 0)

class TestIMSDataBase(unittest.TestCase):

    def test_build_ims_database(self):
        from dataset_management.ims_dataset.add_raw_data_to_database import test
        r = test()


class TestProcessed(unittest.TestCase):

    def test_build_processed(self):
        from dataset_management.general.update_database_with_processed import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)


class TestAugment(unittest.TestCase):

    def test_build_augmented(self):
        from dataset_management.general.update_database_with_augmented import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)

class TestModel(unittest.TestCase):

    def test_build_trained_models(self):
        from dataset_management.general.update_database_with_models import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)

class TestEncode(unittest.TestCase):

    def test_build_encoding(self):
        from dataset_management.general.update_database_with_encodings import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)


class TestMetrics(unittest.TestCase):
    def test_metrics(self):
        from dataset_management.general.update_database_with_metrics import main
        r = main()
        self.assertTrue(r.find_one({}) is not None)


if __name__ == '__main__':
    unittest.main()
