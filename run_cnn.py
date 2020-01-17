
# IMPORTs
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import os
from datetime import datetime
import matplotlib.pyplot as plt
import cnn
from utils import load_data


################################################################################
# print("TF version: {}".format(tf.__version__))
# print("GPU available: {}".format(tf.test.is_gpu_available()))


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

################################################################################
# HYPERPARAMETERS AND DESIGN CHOICES
num_neurons = 128
batch_size = 64
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)
num_epochs = 50
learn_rate = 0.001
drop_prob = 0.1
optim = "Adam"

# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
tb_callback = TensorBoard(log_dir=folder)

# Build, train, and test model
model = cnn.build_model(input_shape, activation_fn, learn_rate, drop_prob, num_neurons, num_classes)
train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model(model, train_images, train_labels,
                                                                         batch_size,num_epochs, valid_images,
                                                                         valid_labels, tb_callback)
test_accuracy, test_loss, predictions = cnn.test_model(model, test_images, test_labels)
#
# # save test set results to csv
# predictions = np.round(predictions)
# predictions = predictions.astype(int)
# df = pd.DataFrame(predictions)
# df.to_csv("mnist.csv", header=None, index=None)
#
# ################################################################################
# # Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)
#
# Loss curves
plt.figure(1)
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves " + optim)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(optim + '_loss.png')
plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves " + optim)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(optim + '_acc.png')
plt.show()
#
# # Test loss and accuracy
# print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
# print("##########")