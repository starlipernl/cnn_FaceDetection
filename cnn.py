################################################################################
# IMPORTs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


################################################################################
# assigns the specified activation function
def get_activ_fn(s):
    if s == "relu":
        return relu
    elif s == "softmax":
        return softmax
    elif s == "tanh":
        return tanh
    elif s == "sigmoid":
        return sigmoid


################################################################################
# build model
def build_model(input_shape, activation_fn, lrn_rate, drop_prob, num_neurons, num_classes):
    model = Sequential()
    model.add(Conv2D(6, 5, input_shape=input_shape, padding="same",
                     activation=activation_fn))
    model.add(MaxPool2D(2, strides=2))
    model.add(Conv2D(16, 5, padding="same", activation=activation_fn))
    model.add(MaxPool2D(2, strides=2))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation=activation_fn))
    model.add(Dropout(drop_prob))
    model.add(Dense(num_classes, activation=softmax))
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=lrn_rate, decay=0.001),
                  metrics=["accuracy"])
    # model.summary()
    return model


############################################################################
# train model
def train_model(model, train_images, train_labels, BATCH_SIZE, NUM_EPOCHS, valid_images, valid_labels, tb_callback):
    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(valid_images, valid_labels),
        shuffle=True,
        callbacks=[tb_callback],
        verbose=2
    )
    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    return train_accuracy, train_loss, valid_accuracy, valid_loss


#################################################################################
# evaluation on test set
def test_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(
        x=test_images,
        y=test_labels,
        verbose=0
    )
    # predictions with test set
    predictions = model.predict_proba(
        x=test_images,
        batch_size=None,
        verbose=0
    )
    return test_accuracy, test_loss, predictions



