import random, time
import numpy as np
import os
import math
from video import Video
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
            
            positive_bag = self.__create_bags(self.train_anom, 1)
            print("Length:",positive_bag.shape[0])
            split_pos = np.array_split(positive_bag, math.ceil(positive_bag.shape[0]/16))
            print("done.")
            for i in range(len(split_pos)):
              print("FRAMES SHAPE: %s" % (str(split_pos[i].shape)))
              bag, _ = self.c3d_model.predict(split_pos[i])
              split_pos[i] = bag
              
            negative_bag = self.__create_bags(self.train_norm, 1)
            split_neg = np.array_split(negative_bag, math.ceil(negative_bag.shape[0]/16))
            for i in range(len(split_neg)):
              bag, _ = self.c3d_model.predict(split_neg[i])
              split_neg[i] = bag
            
            final_pos = np.vstack(split_pos)
            final_pos = final_pos if len(final_pos)%32 == 0 else final_pos[:-(len(final_pos)%32)]
            
            final_neg = np.vstack(split_neg)
            final_neg = final_neg if len(final_neg)%32 == 0 else final_neg[:-(len(final_neg)%32)]

            cost = self.bc_model.train(final_pos, final_neg)
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
        count = 0
        test_labels = []
        predictions = []
        
        start = time.time()
        for row in self.dataset.testing:
            count += 1
            print("Loading video: %s (%d of %d)..." % (row[0], count, len(self.dataset.testing)))
            video = Video(row[0], row[1])
            video.resize(self.img_width, self.img_height)
            frames = np.array(video.getSegments())
            print("FRAMES SHAPE: %s" % (str(frames.shape)))
            if frames.shape[0] == 0:
              continue
            features, _ = self.c3d_model.predict(frames)
            print("FEATURES SHAPE: %s" % (str(features.shape)))
            scores, _ = self.bc_model.predict(features)
            predictions.append(max(scores))
            test_labels.append(int(row[1]))
            
        end = time.time()

        print("TRUE LABELS:")
        print(test_labels)
        print("SCORES:")
        print([p[0] for p in predictions])
        print("PREDICTIONS:")
        print([round(p[0]) for p in predictions])

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
            if t[1] == '0':
                self.train_norm.append(t)
            else:
                self.train_anom.append(t)

    def __create_bags(self, video_collection, total_bags=1):
        bags = []
        count = 0
        for current_bag in range(total_bags):
            count += 1
            video = None
            while True:
              row = random.choice(video_collection)
              print("Loading video: %s..." % (row[0]))
              video = Video(row[0],row[1])
              video.resize(self.img_width, self.img_height)
              instances = video.getSegments()
              print("FINISHED!!!")
              print("Loaded video shape: %s" % (str(instances.shape)))
              if instances.shape[0] > 0:
                break
            bags.append(instances)
        bags = np.vstack(bags)
        print("Loaded video of shape: %s" % (str(bags.shape)))
        return bags

    def __save_interval(self):
        dir = self.model_dir + "/intervals"
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.bc_model.save_model(dir + "/save-" + ("%05d/model"%self.iter))
        self.iter += 1
