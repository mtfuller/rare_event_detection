import random, time
import numpy as np
import os
import math
from video import Video
from models.c3d_model import C3DModel
from models.bc_model import BCModel
from sklearn.metrics import roc_curve, auc

class Algorithm:
    """The Algorithm class contains a dataset object, C3D model, and a Binary Classifier model in order to facilitate
    training and testing of the mode.
    """
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
        # Give default values for total epochs and save interval
        total_epoch = kwargs.get('total_epoch', 10)
        save_interval = kwargs.get('save_interval', 10)

        # Setup training
        cost_curve = []
        epochs_until_save = 0
        begin_train = time.time()

        # Train the model for the given amount of epochs
        for epoch in range(total_epoch):
            # Start timer
            print("Training EPOCH #%d..." % (epoch))
            begin_epoch = time.time()

            # Get positive video batch
            positive_bag = self.__create_bags(self.train_anom, 1)
            split_pos = np.array_split(positive_bag, math.ceil(positive_bag.shape[0]/16))
            for i in range(len(split_pos)):
                print("FRAMES SHAPE: %s" % (str(split_pos[i].shape)))
                bag, _ = self.c3d_model.predict(split_pos[i])
                split_pos[i] = bag
            final_pos = np.vstack(split_pos)
            final_pos = final_pos if len(final_pos)%32 == 0 else final_pos[:-(len(final_pos)%32)]

            # Get negative video batch
            negative_bag = self.__create_bags(self.train_norm, 1)
            split_neg = np.array_split(negative_bag, math.ceil(negative_bag.shape[0]/16))
            for i in range(len(split_neg)):
                bag, _ = self.c3d_model.predict(split_neg[i])
                split_neg[i] = bag
            final_neg = np.vstack(split_neg)
            final_neg = final_neg if len(final_neg)%32 == 0 else final_neg[:-(len(final_neg)%32)]

            # Run training on the positive and negative batch
            cost = self.bc_model.train(final_pos, final_neg)

            # End epoch timer
            end_epoch = time.time()
            print("Finished. Cost: %.5f. Time Elapsed: %.5f sec." % (cost,(end_epoch-begin_epoch)))

            # Add cost to curve
            cost_curve.append(cost)

            # Save the model every "save_interval" times
            if epochs_until_save <= 0:
                self.__save_interval()
                epochs_until_save = save_interval
            else:
                epochs_until_save -= 1

        # Return cost curve and training duration
        end_train = time.time()
        return cost_curve, (end_train-begin_train)

    def test(self):
        # Setup testing
        count = 0
        test_labels = []
        predictions = []

        # Start testing timer
        start = time.time()

        # Get predictions for each video in the testing set
        for row in self.dataset.testing:
            count += 1

            # Load each video
            print("Loading video: %s (%d of %d)..." % (row[0], count, len(self.dataset.testing)))
            video = Video(row[0], row[1])
            video.resize(self.img_width, self.img_height)
            frames = np.array(video.getSegments())
            print("FRAMES SHAPE: %s" % (str(frames.shape)))
            if frames.shape[0] == 0:
              continue

            # Predict the C3D features for the video frames
            features, _ = self.c3d_model.predict(frames)
            print("FEATURES SHAPE: %s" % (str(features.shape)))

            # Predict the anomaly score for the given C3D features
            scores, _ = self.bc_model.predict(features)

            # Add the prediction and true label scores
            predictions.append(max(scores))
            test_labels.append(int(row[1]))

        # Stop the testing timer
        end = time.time()

        # Display results
        print("TRUE LABELS:")
        print(test_labels)
        print("SCORES:")
        print([p[0] for p in predictions])
        print("PREDICTIONS:")
        print([round(p[0]) for p in predictions])

        # Return the FPR, TPR, and AUC for the ROC Curve, as well as the testing time
        _fpr, _tpr, _ = roc_curve(test_labels, predictions)
        _auc = auc(_fpr, _tpr)
        return _fpr, _tpr, _auc, (end-start)

    def save_model(self):
        self.bc_model.save_model(self.model_dir + '/model')

    def load_model(self):
        self.bc_model.load_model(self.model_dir + '/model')

    def __split_train(self):
        # Split the trainig dataset into a collection of positive videos and a collection of negative videos
        train = self.dataset.training
        self.train_norm = []
        self.train_anom = []
        for t in train:
            if t[1] == '0':
                self.train_norm.append(t)
            else:
                self.train_anom.append(t)

    def __create_bags(self, video_collection, total_bags=1):
        # Create an empty bag collection
        bags = []
        count = 0

        # Load and collect a number of bags from the given video collection
        for current_bag in range(total_bags):
            count += 1
            video = None

            # Continually load videos until it finds a video that contains a valid number of frames
            while True:
                # Randomly choose video from video collection
                row = random.choice(video_collection)

                # Load the given video
                print("Loading video: %s..." % (row[0]))
                video = Video(row[0],row[1])
                video.resize(self.img_width, self.img_height)
                instances = video.getSegments()
                print("FINISHED!!!")
                print("Loaded video shape: %s" % (str(instances.shape)))

                # Exit out of the loop if the video is valid
                if instances.shape[0] > 0:
                    break

            # Add bag to collection
            bags.append(instances)

        # Create Numpy array
        bags = np.vstack(bags)
        print("Loaded video of shape: %s" % (str(bags.shape)))
        return bags

    def __save_interval(self):
        # Saves the current model
        dir = self.model_dir + "/intervals"
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.bc_model.save_model(dir + "/save-" + ("%05d/model"%self.iter))
        self.iter += 1
