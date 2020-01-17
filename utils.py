import os
import cv2
import numpy as np
import random


# function to load list of images from a specified directory
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 1)
        img_flat = img.flatten()
        #img_norm = img/np.float32(255)
        images.append(img_flat)
    return images


# subtract training mean and divide by training standard deviation (standardization)
def standard(train_img, val_img, test_img):
    train_mean = np.mean(train_img, axis=0)
    train_std = np.std(train_img, axis=0)
    train_img_std = np.reshape(np.divide((train_img - train_mean), train_std), (len(train_img), 60, 60, 3))
    valid_img_std = np.reshape(np.divide((val_img - train_mean), train_std), (len(val_img), 60, 60, 3))
    test_img_std = np.reshape(np.divide((test_img - train_mean), train_std), (len(test_img), 60, 60, 3))
    return train_img_std, valid_img_std, test_img_std


# function to load all data in train/test split for both face and nonface categories
def load_data():
    # setup file paths
    face_train_path = os.path.join("Extracted_Faces", "Train")
    face_test_path = os.path.join("Extracted_Faces", "Test")
    non_train_path = os.path.join("Extracted_Non", "Train")
    non_test_path = os.path.join("Extracted_Non", "Test")
    # load images
    face_train = load_images(face_train_path)
    face_test = load_images(face_test_path)
    non_train = load_images(non_train_path)
    non_test = load_images(non_test_path)
    # create label vector
    train_labels = np.append(np.zeros((10000, 1), dtype=int), np.ones((10000, 1), dtype=int))
    test_labels = np.append(np.zeros((1000, 1), dtype=int), np.ones((1000, 1), dtype=int))
    # combine training and testing sets with labels to randomly shuffle the data
    train_data = non_train + face_train
    test_data = non_test + face_test
    train = list(zip(train_data, train_labels))
    test = list(zip(test_data, test_labels))
    random.shuffle(train)
    random.shuffle(test)
    train_data, train_labels = zip(*train)
    test_data, test_labels = zip(*test)
    # training validation split 90/10
    train_data_split = np.asarray(train_data[0:18000])
    eval_data = np.asarray(train_data[18000:-1])
    test_data = np.asarray(test_data)
    train_labels_split = np.asarray(train_labels[0:18000])
    eval_labels = np.asarray(train_labels[18000:-1])
    test_labels = np.asarray(test_labels)
    # Standardize the data
    train_data_std, eval_data_std, test_data_std = standard(train_data_split, eval_data, test_data)
    return train_data_std, train_labels_split, eval_data_std, eval_labels, test_data_std, test_labels
