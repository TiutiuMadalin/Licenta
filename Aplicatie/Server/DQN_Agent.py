import numpy as np
import random
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input

class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.gamma = 0.8

        self.tau = 0.01
        self.batch_size = 100
        self.memory = deque(maxlen=2000)
        #self.double_dqn = False
        self.input_, self.model = self.create_model()
        _, self.target_model = self.create_model()

    def create_model(self):
        input_ = Input(shape=(self.state_size,))
        hidden_layer_1 = Dense(units=128, activation='relu', input_shape=(self.state_size,))(input_)
        hidden_layer_2 = Dense(units=64, activation='relu', input_shape=(128,))(hidden_layer_1)
        hidden_layer_3 = Dense(units=32, activation='relu', input_shape=(64,))(hidden_layer_2)
        output = Dense(units=self.action_size, activation='linear', input_shape=(32,))(hidden_layer_3)
        model = Model(inputs=input_, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        # return value of each action
        return input_, model

    def remember(self, cur_state, action, next_state, reward, done):
        self.memory.append([cur_state, action, next_state, reward, done])

    def act(self, cur_state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #if np.random.rand() < self.epsilon:
            #return np.random.randint(0, 2)
        predicted_action = self.model.predict(cur_state)
        return np.argmax(predicted_action[0])

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        for cur_state, action_, next_state, reward_, done in mini_batch:
            target = self.model.predict(cur_state)
            if not done:
                if self.double_dqn:
                    predicted_action = np.argmax(self.model.predict(next_state)[0])
                    target_q = self.target_model.predict(next_state)[0][predicted_action]
                    target[0][action_] = reward_ + self.gamma * target_q
                else:
                    target_q = self.target_model.predict(next_state)[0]
                    target[0][action_] = reward_ + self.gamma * max(target_q)

            else:
                target[0][action_] = reward_

            self.model.fit(cur_state, target, verbose=0)

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save(self, file):
        self.model.save_weights(file)

    def load(self, file):
        self.model.load_weights(file)
        self.target_model.load_weights(file)
