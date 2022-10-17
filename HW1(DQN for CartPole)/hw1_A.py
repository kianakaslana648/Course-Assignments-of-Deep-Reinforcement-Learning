import gym
from agent import DQN
import matplotlib.pyplot as plt
import random
import numpy as np

class ReplayMemory(object):
    def __init__(self):
        self.MemoryPool = []
        self.maxLength = 10000
        self.pointer = 0

    def push(self, s_old, action, s_new, reward):
        if len(self.MemoryPool) < self.maxLength:
            self.MemoryPool.append([s_old, action, s_new, reward])
        else:
            self.MemoryPool[self.pointer] = [s_old, action, s_new, reward]
            self.pointer = (self.pointer + 1) % self.maxLength

    def sample(self):
        return random.choice(self.MemoryPool)



def main(args):
    all_return_train = []
    all_return_test = []
    for i in range(len(args.alpha)):
        alpha = args.alpha[i]
        max_Time = 10000
        step_of_target_network_update = 2
        test_episodes = 100
        env = gym.make(args.env)
        buffer = ReplayMemory()
        agent = DQN(num_of_actions=env.action_space.n, num_of_observations=env.observation_space.shape[0])
        agent.alpha = alpha
        bootstrapping_step = 2
        #########
        ######### train train train
        return_train = []

        print('''Alpha={}'''.format(alpha))
        for epoch in range(args.epoch):
            # show epoch
            if(args.isEpochPrinted):
                print('''Epoch:{}/{}'''.format(epoch+1, args.epoch))
            # Initialize the Sequence
            cur_state = env.reset()[0]
            undiscounted_reward_sum = 0
            # Start an Episode
            for time in range(max_Time):
                # show time
                if (args.isTimePrinted):
                    print('''   Time:{}/{}'''.format(time + 1, max_Time))
                cur_action = agent.forward(cur_state)
                next_state, reward, terminal = list(env.step(cur_action))[:3]
                reward = reward if not terminal else 0
                buffer.push(cur_state, cur_action, next_state, reward)
                # Estimate Returns
                undiscounted_reward_sum = undiscounted_reward_sum + reward
                # Improve Policy
                for _ in range(bootstrapping_step):
                    update_input = buffer.sample()
                    agent.update_Q_network(update_input)
                # whether terminate or not
                if terminal:
                    print('''   TotalTime:{}'''.format(time + 1))
                    break
                # update current state
                cur_state = next_state
                # update target network
                if (time+1) % step_of_target_network_update == 0:
                    agent.store_Q_network()
            return_train.append(undiscounted_reward_sum)
            # Verbose
            if args.isVerbose:
                if (epoch + 1) % 50 == 0:
                    plt.plot(return_train, color='blue')
                    plt.xlabel('Episodes')
                    plt.ylabel('Returns')
                    plt.title('''Return Stats: Epoch {}'''.format(epoch + 1))
                    plt.show()
        all_return_train.append(return_train)

        #print(agent.model.state_dict())
        #exit()
        #########
        ######### test test test
        # return_test = []
        # for epoch in range(test_episodes):
        #     # Show episodes
        #     if (args.isEpochPrinted):
        #         print('''Test Epoch:{}/{}'''.format(epoch + 1, test_episodes))
        #     # Initialize the Sequence
        #     cur_state = env.reset()[0]
        #     undiscounted_reward_sum = 0
        #     for time in range(max_Time):
        #         # show time
        #         if (args.isTimePrinted):
        #             print('''   Time:{}/{}'''.format(time + 1, max_Time))
        #         cur_action = agent.forward(cur_state)
        #         next_state, reward, terminal = list(env.step(cur_action))[:3]
        #         reward = reward if not terminal else 0
        #         # Estimate Returns
        #         undiscounted_reward_sum = undiscounted_reward_sum + reward
        #         # whether terminate or not
        #         if terminal:
        #             print('''   TotalTime:{}'''.format(time + 1))
        #             break
        #         # update current state
        #         cur_state = next_state
        #
        #     return_test.append(undiscounted_reward_sum)

        #all_return_test.append(return_test)
        env.close()

    return all_return_train, all_return_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v0")
    parser.add_argument("--epoch", default=500, type=int)
    parser.add_argument("--alpha", default='0.01', type=str)
    parser.add_argument("--isEpochPrinted", default=True, type=bool)
    parser.add_argument("--isTimePrinted", default=False, type=bool)
    parser.add_argument("--isVerbose", default=False, type=bool)

    args = parser.parse_args()
    args.alpha = [float(temp_alpha) for temp_alpha in args.alpha.split()]
    train_stats, test_stats = main(args)

    stats = train_stats

    plt.clf()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    for i in range(len(stats)):
        one_return = stats[i]
        ax.plot(one_return, label='''Alpha={}'''.format(args.alpha[i]))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    ax.title.set_text('''Return Stats''')
    ax.legend()
    if len(args.alpha) == 1:
        fig.savefig('Return Stats.png', facecolor='w')
    else:
        fig.savefig('Return Stats with Different Alphas.png', facecolor='w')
    plt.show()