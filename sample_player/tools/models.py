from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model


def new_dqn_model():
    input_n = Input(shape=(88,), name="input")

    x = Dense(176, activation='tanh')(input_n)
    x = Dense(352, activation='tanh')(x)
    x = Dense(88, activation='tanh')(x)
    out = Dense(3)(x)
    model = Model(inputs=[input_n], outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


def push_or_fold_model():

    input_n = Input(shape=(16,), name="input")

    x = Dense(32, activation='relu')(input_n)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(2)(x)
    model = Model(inputs=[input_n], outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


def dqn_model():

    input_n = Input(shape=(16,), name="input")

    x = Dense(32, activation='relu')(input_n)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(3)(x)
    model = Model(inputs=[input_n], outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def save_model(model, name):
    model.save(name)


def load(name):
    return load_model(name)

