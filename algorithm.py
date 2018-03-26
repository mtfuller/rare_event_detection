import random, time
import numpy as np
import os
from model import Model


class Algorithm:
    def __init__(self, **kwargs):
        self.dataset = kwargs.get('dataset', None)
        self.clip_length = kwargs.get('clip_length', 4)
        self.img_width = kwargs.get('img_width', 112)
        self.img_height = kwargs.get('img_height', 112)
        self.__split_train()
        self.model_dir = kwargs.get('model_dir', "./models")
        self.model = Model()
        self.iter = 0

    def build(self):
        self.model.build()

    def train(self, **kwargs):
        total_epoch = kwargs.get('total_epoch', 10)
        save_interval = kwargs.get('save_interval', 10)

        cost_curve = []
        epochs_until_save = 0
        begin_train = time.time()
        for epoch in range(total_epoch):
            print("Training EPOCH #%d..." % (epoch))
            begin_epoch = time.time()
            positive_bag = self.__create_bag(self.train_anom)
            negative_bag = self.__create_bag(self.train_norm)
            cost = self.model.train(positive_bag, negative_bag)
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
        test_videos = [video.getFrames() for video in self.dataset.testing]
        test_labels = [int(video.getAnomaly()) for video in self.dataset.testing]

        bags = []
        for video in test_videos:
            bag = []
            num_of_segments = int(len(video) / 4)
            frame_index = 0
            for i in range(num_of_segments):
                segment = np.array([video[frame_index]])
                frame_index += 1
                for j in range(3):
                    segment = np.vstack((segment, [video[frame_index]]))
                    frame_index += 1
                bag.append(segment)
            bags.append(bag)

        predictions = []
        start = time.time()
        for video in bags:
            video = np.array(video)
            scores, _ = self.model.predict(video)
            predictions.append(max(scores))
        end = time.time()

        return test_labels, predictions, (end-start)


    def save_model(self):
        self.model.saveModel(self.model_dir + '/model')

    def load_model(self, dir):
        self.model.loadModel(dir + '/model')

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

    def __create_bag(self, video_collection):
        video = random.choice(video_collection)
        video_frames = video.getFrames()
        num_of_segments = int(len(video_frames)/4)
        frame_index = 0
        bag = []
        for i in range(num_of_segments):
            segment = np.array([video_frames[frame_index]])
            frame_index += 1
            for j in range(3):
                segment = np.vstack((segment, [video_frames[frame_index]]))
                frame_index += 1
            bag.append(segment)
        return bag


    def __save_interval(self):
        dir = self.model_dir + "/intervals"
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.model.saveModel(dir + "/save-" + ("%05d/model"%self.iter))
        self.iter += 1