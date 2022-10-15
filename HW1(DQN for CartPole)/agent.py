import tensorflow as tf
import numpy as np
import random

class DQN:
    # define your DQN agent network
    def __init__(self, num_of_actions, num_of_observations, num_of_hidden_layers=3,\
            num_of_elements_per_layer=-1, epsilon=0.1, gamma=1.0, alpha=0.0001, isDebug=False):
        print('### Initialize DQN')
        self.num_of_actions = num_of_actions
        self.num_of_observations = num_of_observations
        self.action_space = np.arange(num_of_actions)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.isDebug = isDebug
        if(num_of_elements_per_layer == -1):
            self.num_of_elements_per_layer = num_of_observations
        else:
            self.num_of_elements_per_layer = num_of_elements_per_layer

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.num_of_observations,)))
        for _ in range(num_of_hidden_layers):
            self.model.add(tf.keras.layers.Dense(self.num_of_elements_per_layer, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.num_of_actions, activation='sigmoid'))
        self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
        self.model.set_weights([np.zeros(temp_element.shape) for temp_element in self.model.get_weights()])

        self.model_target = tf.keras.Sequential()
        self.model_target.add(tf.keras.layers.Input(shape=(self.num_of_observations,)))
        for _ in range(num_of_hidden_layers):
            self.model_target.add(tf.keras.layers.Dense(self.num_of_elements_per_layer, activation='relu'))
        self.model_target.add(tf.keras.layers.Dense(self.num_of_actions, activation='sigmoid'))
        self.model_target.compile(optimizer='adam', loss='categorical_crossentropy')

    # copy weights from current Q-network into target network
    def store_Q_network(self):
        if self.isDebug:
            print('### Store Q Network')
        self.model_target.set_weights(self.model.get_weights())

    # update parameters of Q-network
    def update_Q_network(self, input):
        if self.isDebug:
            print('### Update Q Network')
        cur_state, action, next_state, reward = input
        Q_values_target = self.model_target.predict(np.array([cur_state])).flatten()
        Q_values = self.model.predict(np.array([cur_state])).flatten()
        y = reward + self.gamma * np.max(Q_values_target)
        with tf.GradientTape() as tape:
            res_X = tf.Variable([cur_state])
            res_y = self.model(res_X)[:, action]
            res_w = self.model.weights
            res_grads = tape.gradient(res_y, res_w)
        new_w = [np.array(res_w[i]-self.alpha*(Q_values[action]-y)*res_grads[i]) for i in range(len(res_grads))]
        self.model.set_weights(new_w)

    # call to determine next action (epsilon-greedy)
    def forward(self, cur_state):
        if self.isDebug:
            print('### Forward')
        Q_values = self.model.predict(np.array([cur_state])).flatten()
        best_action = np.argmax(Q_values)
        temp_prob = np.full(self.num_of_actions, self.epsilon/self.num_of_actions)
        temp_prob[best_action] += 1 - self.epsilon
        next_action = random.choices(self.action_space, weights=temp_prob, k=1)[0]
        return next_action