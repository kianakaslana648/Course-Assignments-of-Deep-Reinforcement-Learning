import torch
import numpy as np
import random
import copy

class DQN:
    # define your DQN agent network
    def __init__(self, num_of_actions, num_of_observations, num_of_hidden_layers=2,\
            num_of_elements_per_layer=64, epsilon=0.2, gamma=1.0, alpha=0.0001, isDebug=False):
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


        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_of_observations, num_of_elements_per_layer, bias=False),
            torch.nn.ELU(),
            torch.nn.Linear(num_of_elements_per_layer, num_of_elements_per_layer * 2, bias=False),
            torch.nn.ELU(),
            torch.nn.Linear(num_of_elements_per_layer * 2, num_of_elements_per_layer * 2, bias=False),
            torch.nn.ELU(),
            torch.nn.Linear(num_of_elements_per_layer * 2, num_of_actions, bias=False),
            torch.nn.Flatten(0,1)
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.NAdam(self.model.parameters(), self.alpha)

        self.model_target = copy.deepcopy(self.model)
        # self.model = torch.nn.Sequential()
        # self.model.add_module('Input',torch.nn.Linear(self.num_of_observations, self.num_of_elements_per_layer))
        # self.model.add_module('Input_ELU',torch.nn.ELU())
        # for i in range(num_of_hidden_layers):
        #     self.model.add_module("Linear{}".format(i),torch.nn.Linear(self.num_of_elements_per_layer, self.num_of_elements_per_layer))
        #     self.model.add_module("ELU{}".format(i),torch.nn.ELU())
        # self.model.add_module("Output",torch.nn.Linear(self.num_of_elements_per_layer, self.num_of_actions))
        # self.model.add_module("Output_Flatten",torch.nn.Flatten(0,1))
        #
        # self.model_target = torch.nn.Sequential()
        # self.model_target.add_module('Input',torch.nn.Linear(self.num_of_observations, self.num_of_elements_per_layer))
        # self.model_target.add_module('Input_ELU',torch.nn.ELU())
        # for i in range(num_of_hidden_layers):
        #     self.model_target.add_module("Linear{}".format(i),torch.nn.Linear(self.num_of_elements_per_layer, self.num_of_elements_per_layer))
        #     self.model_target.add_module("ELU{}".format(i),torch.nn.ELU())
        # self.model_target.add_module("Output",torch.nn.Linear(self.num_of_elements_per_layer, self.num_of_actions))
        # self.model_target.add_module("Output_Flatten",torch.nn.Flatten(0,1))
        self.model_target.load_state_dict(self.model.state_dict())

    # copy weights from current Q-network into target network
    def store_Q_network(self):
        if self.isDebug:
            print('### Store Q Network')
        self.model_target.load_state_dict(self.model.state_dict())

    # update parameters of Q-network
    def update_Q_network(self, input):
        if self.isDebug:
            print('### Update Q Network')
        cur_state, action, next_state, reward = input
        Q_values_target = self.model_target(torch.tensor(np.array([cur_state])).float())

        #self.optimizer.zero_grad()
        Q_values = self.model(torch.tensor(np.array([cur_state])).float())
        y = reward + self.gamma * (max(Q_values_target).item())
        #print('y:'+str(y))

        loss = self.criterion(Q_values[action], torch.Tensor([y]))


        #loss = (Q_values[action] - y)**2
        #self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        #NoNan = True
        # for param in self.model.parameters():
        #     if not torch.isfinite(param.grad).all().item():
        #         NoNan = False
        #         break

        # if NoNan:
        #     with torch.no_grad():
        #         for param in self.model.parameters():
        #             param -= (self.alpha * (Q_values[action].item() - y)) * param.grad
        # if NoNan:
        #     with torch.no_grad():
        #         for param in self.model.parameters():
        #             #print(param.grad)
        #             param -= self.alpha * param.grad

    # call to determine next action (epsilon-greedy)
    def forward(self, cur_state):
        if self.isDebug:
            print('### Forward')
        with torch.no_grad():
            Q_values = self.model(torch.tensor(np.array([cur_state])).float())
        best_action = Q_values.detach().numpy().argmax()
        temp_prob = np.full(self.num_of_actions, self.epsilon/self.num_of_actions)
        temp_prob[best_action] += 1 - self.epsilon
        next_action = random.choices(self.action_space, weights=temp_prob, k=1)[0]
        return next_action

    # call to determine next action (argmax)
    def predict(self, cur_state):
        if self.isDebug:
            print('### Predict')
        Q_values = self.model(torch.tensor(np.array([cur_state])).float())
        best_action = Q_values.detach().numpy().argmax()
        return best_action