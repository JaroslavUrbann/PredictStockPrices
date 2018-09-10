import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import random
import sys
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = []
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.975

        if is_eval:
            self.model = load_model("models/" + model_name)
        else:
            self.model = self._model()

    def _model(self):
        model = Sequential()
        model.add(LSTM(64, batch_input_shape=(1, self.state_size, 1), stateful=True, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(2, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer="adam")

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state.reshape(1, self.state_size, 1))
        return np.argmax(options[0])

    def expReplay(self):
        for state, action, reward, next_state, done in self.memory:
            target = reward
            if not done:
                target = reward * self.gamma * np.amax(
                    self.model.predict(next_state.reshape(1, self.state_size, 1))[0])
            target_f = self.model.predict(state.reshape(1, self.state_size, 1))
            target_f[0][action] = target
            # print("Reward: " + str(reward) + "   |   Target: " + str(target))
            self.model.train_on_batch(state.reshape(1, self.state_size, 1), target_f)

        self.memory = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay