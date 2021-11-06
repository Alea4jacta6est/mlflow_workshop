import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers

from src.preprocess import get_processed_data


def create_keras_model(num_classes: int):
    """Adds layers to keras CNN, configures
    optimizers, loss type and metrics

    Args:
        num_classes (int): classes number

    Returns:
        model (model object)
    """
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def train_and_save(num_classes: int, batch_size: int, epochs: int):
    """Trains created model with given number of epochs, batch size etc.

    Args:
        num_classes (int): classes number
        batch_size (int): batch size
        epochs (int): epochs number
    """
    model_name = f'models/keras_mnist_model.h5'
    x_train, y_train, x_test, y_test = get_processed_data()
    model = create_keras_model(num_classes)
    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))
    print("The model has been successfully trained")
    model.save(model_name)
    print(f"Saving the model as {model_name}")
    return model, model_name


if __name__ == "__main__":
    train_and_save()
