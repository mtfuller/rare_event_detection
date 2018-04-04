import unittest, shutil, os
from ucsd_dataset import ucsd_dataset
from algorithm import Algorithm

MODEL_DIR = "./models/MY_TEST_MODEL"

class TestAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ucsd_dataset(pedestrian="1")
        cls.myNewAlgorithm = Algorithm(dataset=cls.dataset, model_dir=MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        if (os.path.isdir(MODEL_DIR)):
            shutil.rmtree(MODEL_DIR)

    def test_train(self):
        algo = self.__class__.myNewAlgorithm
        total_epochs = 2
        cost_curve, time = algo.train(total_epoch=total_epochs, save_interval=1)
        self.assertTrue(isinstance(cost_curve, list) and len(cost_curve) == total_epochs,
                        "The cost curve is not a valid list: %s" % (str(cost_curve)))
        self.assertTrue(isinstance(time, float) and time > 0.0,
                        "The value for time is not correct: %f" % (time))

    def test_test(self):
        algo = self.__class__.myNewAlgorithm
        fpr, tpr, auc, time = algo.test()
        self.assertTrue(isinstance(time, float) and time > 0.0,
                     "The value for time is not valid: %f" % (time))

    def test_save_load(self):
        algo = self.__class__.myNewAlgorithm
        fpr, tpr, auc1, _ = algo.test()
        algo.save_model()
        algo2 = Algorithm(dataset=self.__class__.dataset, model_dir=MODEL_DIR)
        algo2.load_model()
        fpr, tpr, auc2, _ = algo2.test()
        self.assertEqual(auc1, auc2, "Loaded algorithm test had an AUC of %f, but should have been %f" % (auc2, auc1))
        print("new model has AUC of %f and old model had %f" % (auc2, auc1))
