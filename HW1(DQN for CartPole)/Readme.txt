hw1_A.py:

Builds a DQN model and trains it with the env of CartPole within 500 episodes.

Saves a plot for the estimated log-returns on episodes.

Parameters:

--alpha: str
Default: '0.1'
Creates models for different alphas. 
Example cmd line:
python hw1_A.py --alpha "0.01 0.001 0.0001 0.00001"
After executing the cmd line above, u have one final plot of log-stats for these 4 alphas.

--epoch: int
Default: 500
Controls total number of episodes for training.

--env: str
Default: 'CartPole-v0'
U'd better not change it cauz this DQN model basically works for CartPole.

--isEpochPrinted: bool
Default: True
Whether or not print the current epoch.

--isBTPrinted: bool
Default: False
Whether or not print the current bootstrapping step.

--isVerbose: bool
Default: False
Whether or not show a plot of log-stats every 50 episodes.

Dependencies:
agent.py: tensorflow, numpy, random, 
hw1_A.py: gym, agent, matplotlib, random, numpy