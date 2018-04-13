from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers import Dense, Flatten, Conv2D


def create_model():
    # The first layer in a Keras network always needs to know the size of its input
    # For other layers, this is calculated automatically
    # input = Input(shape=(nr_channels * image_size * image_size,))
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(9,16), strides = (9, 16), padding='same',
                  data_format='channels_first', activation='relu', input_shape=(12,288,512)))
    model.add(Conv2D(filters=64, kernel_size=(8,8), strides = 4, padding='same',
                  data_format='channels_first', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(2,2), strides = 2, padding='same',
                  data_format='channels_first', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=2, activation='linear'))

    model.compile(optimizer='rmsprop',
                  loss='mse')
    return model

def main():
    model = create_model()
    model.predict()
    print(model.summary())

if __name__ == "__main__":
    main()
