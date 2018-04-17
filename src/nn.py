from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(7,4), strides = (7, 4), padding='same',
                  data_format='channels_last', activation='relu', input_shape=(206,144,4)))
    model.add(Conv2D(filters=64, kernel_size=(8,8), strides = 4, padding='same',
                  data_format='channels_last', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(2,2), strides = 2, padding='same',
                  data_format='channels_last', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=2, activation='linear'))

    model.compile(optimizer='rmsprop',
                  loss='mse')
    return model


def main():
    model = create_model()
    print(model.summary())


if __name__ == "__main__":
    main()
