
# IMPORTs
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import set_session
import os
from datetime import datetime
import cnn
from utils import load_data
from random import uniform


################################################################################
# print("TF version: {}".format(tf.__version__))
# print("GPU available: {}".format(tf.test.is_gpu_available()))

# # Limit memory usage when running on titan
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
# set_session(tf.Session(config=config))

################################################################################

# loading dataset
train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

img_rows, img_cols =60, 60
num_channels = 3
input_shape = (img_rows, img_cols, num_channels)

# output dimensions
num_classes = 2

# HYPERPARAMETERS AND DESIGN CHOICES
num_neurons = 128
batch_size = 64
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)
num_epochs = 5
max_count = 50
for count in range(0, max_count):
    learn_rate = 10**uniform(-2, -4)
    drop_prob = 10**uniform(-2, 0)

    # callbacks for Save weights, Tensorboard
    # creating a new directory for each run using timestamp
    folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
    tb_callback = TensorBoard(log_dir=folder)

    # Build, train, and test model
    model = cnn.build_model(input_shape, activation_fn, learn_rate, drop_prob, num_neurons, num_classes)
    train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model(model, train_images, train_labels,
                                                                             batch_size,num_epochs, valid_images,
                                                                             valid_labels, tb_callback)
    print('Step: {:d}/{:d}, learn: {:.6f}, dropout: {:.4f},'
          'Train_loss: {:.4f}, Train_acc: {:.4f}, Val_loss: {:.4f}, Val_acc: {:.4f}'.format(
          count, max_count, learn_rate, drop_prob, train_loss[-1], train_accuracy[-1],
          valid_loss[-1], valid_accuracy[-1]))

