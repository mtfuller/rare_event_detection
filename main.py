import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

MODEL_PATH = "./pretrained_models/c3d_ucf101_finetune_whole_iter_20000_TF.model"

hello = tf.constant('Hello, TensorFlow!')
print("Starting TF session...")
sess = tf.Session()
print("Session Started!")
print("Loading model...")

# Conv 1







sess.run(tf.trainable_variables())
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess, MODEL_PATH)
print("Model loaded.")
print("FINISHED")