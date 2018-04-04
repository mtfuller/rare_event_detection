import random, time
import numpy as np
import os
from models.c3d_model import C3DModel
from models.bc_model import BCModel
from sklearn.metrics import roc_curve, auc


class Algorithm:
    def __init__(self, **kwargs):
        self.dataset = kwargs.get('dataset', None)
        self.clip_length = kwargs.get('clip_length', 4)
        self.img_width = kwargs.get('img_width', 112)
        self.img_height = kwargs.get('img_height', 112)
        self.__split_train()
        self.model_dir = kwargs.get('model_dir', "./models/AlgorithmTest")
        self.c3d_model = C3DModel()
        self.bc_model = BCModel()
        self.iter = 0

    def train(self, **kwargs):
        total_epoch = kwargs.get('total_epoch', 10)
        save_interval = kwargs.get('save_interval', 10)

        cost_curve = []
        epochs_until_save = 0
        begin_train = time.time()
        for epoch in range(total_epoch):
            print("Training EPOCH #%d..." % (epoch))
            begin_epoch = time.time()
            positive_bag = self.__create_bags(self.train_anom, 3)
            negative_bag = self.__create_bags(self.train_norm, 3)
            positive_bag, _ = self.c3d_model.predict(positive_bag)
            negative_bag, _ = self.c3d_model.predict(negative_bag)
            cost = self.bc_model.train(positive_bag, negative_bag)
            end_epoch = time.time()
            print("Finished. Cost: %.5f. Time Elapsed: %.5f sec." % (cost,(end_epoch-begin_epoch)))
            cost_curve.append(cost)
            if epochs_until_save <= 0:
                self.__save_interval()
                epochs_until_save = save_interval
            else:
                epochs_until_save -= 1
        end_train = time.time()
        return cost_curve, (end_train-begin_train)

    def test(self):
        for video in self.dataset.testing:
            video.resize(self.img_width, self.img_height)
        test_videos = [video.getSegments() for video in self.dataset.testing]
        test_labels = [int(video.getAnomaly()) for video in self.dataset.testing]

        predictions = []
        start = time.time()
        for video in test_videos:
            video = np.array(video)
            features, _ = self.c3d_model.predict(video)
            scores, _ = self.bc_model.predict(features)
            predictions.append(max(scores))
        end = time.time()

        print("TRUE LABELS:")
        print(test_labels)
        print("PREDICTIONS:")
        print(predictions)

        _fpr, _tpr, _ = roc_curve(test_labels, predictions)
        _auc = auc(_fpr, _tpr)

        return _fpr, _tpr, _auc, (end-start)

    def save_model(self):
        self.bc_model.save_model(self.model_dir + '/model')

    def load_model(self):
        self.bc_model.load_model(self.model_dir + '/model')

    def __split_train(self):
        train = self.dataset.training
        self.train_norm = []
        self.train_anom = []
        for t in train:
            t.resize(self.img_width, self.img_height)
            if t.getAnomaly() == '0':
                self.train_norm.append(t)
            else:
                self.train_anom.append(t)

    def __create_bags(self, video_collection, total_bags=1):
        bags = []
        for current_bag in range(total_bags):
            video = random.choice(video_collection)
            instances = video.getSegments()
            bags.append(instances)
        bags = np.vstack(bags)
        print("BAG SHAPE:")
        print(bags.shape)
        return bags

    def __save_interval(self):
        dir = self.model_dir + "/intervals"
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.bc_model.save_model(dir + "/save-" + ("%05d/model"%self.iter))
        self.iter += 1
