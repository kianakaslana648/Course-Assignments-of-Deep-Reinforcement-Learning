import gym
from agent import DQN
import matplotlib.pyplot as plt
import random
import numpy as np

class ReplayMemory(object):
    def __init__(self):
        self.MemoryPool = []

    def push(self, s_old, action, s_new, reward):
        self.MemoryPool.append([s_old, action, s_new, reward])

    def sample(self):
        return random.choice(self.MemoryPool)



def main(args):
    log_stats_result = []
    for i in range(len(args.alpha)):
        alpha = args.alpha[i]
        bootstrapping_steps = 5
        log_return = []
        env = gym.make(args.env)
        agent = DQN(num_of_actions=env.action_space.n, num_of_observations=env.observation_space.shape[0])
        agent.alpha = alpha
        buffer = ReplayMemory()
        if i == 0:
            start_state = env.reset()[0]
            cur_state = start_state
        else:
            cur_state = env.reset()[0]
        print('''Alpha={}'''.format(alpha))
        for epoch in range(args.epoch):
            # show epoch
            if(args.isEpochPrinted):
                print('''Epoch:{}/{}'''.format(epoch+1, args.epoch))
            # Collect Trajectories
            cur_action = agent.forward(cur_state)
            next_state, reward, terminal = list(env.step(cur_action))[:3]
            reward = reward if not terminal else 0
            buffer.push(cur_state, cur_action, next_state, reward)
            agent.store_Q_network()
            # Estimate Returns
            estimated_return = np.max(agent.model.predict(np.array([start_state])).flatten())
            # Improve Policy
            for bt_step in range(bootstrapping_steps):
                # show bootstrapping step
                if(args.isBTPrinted):
                    print('''\tBootstrapping:{}/{}'''.format(bt_step+1, bootstrapping_steps))
                # update parameters
                update_input = buffer.sample()
                agent.update_Q_network(update_input)
            # Log Statistics
            log_return.append(np.log(estimated_return))
            if args.isVerbose:
                if (epoch+1)%50 == 0:
                    plt.plot(log_return, color='blue')
                    plt.xlabel('Episodes')
                    plt.ylabel('Log Returns')
                    plt.title('''Log Stats: Epoch {}'''.format(epoch+1))
                    plt.show()
            # whether terminate or not
            if terminal:
                next_state = env.reset()[0]
            # update current state
            cur_state = next_state

        env.close()
        log_stats_result.append(log_return)
    return log_stats_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v0")
    parser.add_argument("--epoch", default=500, type=int)
    parser.add_argument("--alpha", default='0.1', type=str)
    parser.add_argument("--isEpochPrinted", default=True, type=bool)
    parser.add_argument("--isBTPrinted", default=False, type=bool)
    parser.add_argument("--isVerbose", default=False, type=bool)

    args = parser.parse_args()
    args.alpha = [float(temp_alpha) for temp_alpha in args.alpha.split()]
    log_stats = main(args)

    plt.clf()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    for i in range(len(log_stats)):
        log_return = log_stats[i]
        ax.plot(log_return, label='''Alpha={}'''.format(args.alpha[i]))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Log Returns')
    ax.title.set_text('''Log Stats''')
    ax.legend()
    if len(args.alpha) == 1:
        fig.savefig('Log Stats.png', facecolor='w')
    else:
        fig.savefig('Log Stats with Different Alphas.png', facecolor='w')