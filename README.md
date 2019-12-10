# Deep reinforcement learning stuff

All the code for this book can be found here: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

## Taxonomy of RL methods

All methods in RL can be classified along various dimensions:
* Model-free vs model-based
* Value-based vs policy-based
* On-policy vs off-policy


## Cross-entropy method
Take “educated” random guesses on actions. Select top performers of guesses and use them to train on. 

The *cross-entropy* method is a _model-free_ _policy-based_ method.

'Model-free' simply means the method doesn't build a model of the environment or reward, it just directly connects observations to actions (or values that are related to actions). In other words, the agent simply takes the current observations and does some computations on them, and the result is the action the agent should take. 'Model-based' on the other hand, tries to predict what the next observation and/or reward will be. Model-based methods are often used in deterministic environments, like board games with strict rules. Model-free methods are often easier to train, but it can be hard for them to work well in complex environments. 

'Policy-based' methods approximate the policy of the agent, i.e. what actions the agent should carry out at every step. Policy is usually represented by a probability distribution over the available actions. In 'value-based' methods, the agent calculates the value of every possible action, and chooses the action with the highest value. 

'Off-policy' is the ability to learn on old historical data.

So, cross-entropy is model-free, policy-based, on-policy. This means it doesn't build a model of the environment (it just says to the agent what to do at every step), it approximates the policy of the agent, and it requires fresh data obtained from the environment. 

The cross-entropy method is simple, but powerful.

It's drawbacks are that:
* For training, our episodes have to be finite and short
* The episodes need enough variability of total reward to separate good and bad episodes
* It doesn't work well when there is no intermediate indication about whether an agent is succeeding or not

## Tabular learning (Q-Learning) and the Bellman Equation
The value of a state is the expected total reward that is obtainable from that state.

The key is that when deciding up on an action, the agent does not simply pick the action that leads to the greatest immediate reward, but rather it looks at which action maximises immediate reward plus the long term value of the state it will land in. With this extension, behaviour becomes optimal. The equation for this (that the value of a state is equal to the sum of the best reward and discounted next state value you can get) is called the Bellman equation (for a deterministic case).

We can generalise this deterministic Bellman equation into a more general case by summing across a probability distribution. The interpretation is still the same - that the optimal value of a state is equal to the maximum possible immediate reward plus the discounted long-term value of the state. 

### The Q-Value
In order to make our life easier, we define an additional quantity in addition to the value of the state (V_s). 

We define this new value as the 'Q value' of an action and state (Q_a,s), i.e. it equals the total reward we can get by executing action a in state s. Q values are slightly more convenient in practice than simple state values, and gave rise to a number of methods based upon them called Q-learning. 

Because V_s is the max Q_a,s for a state, we can define Q in terms of itself, so Q_s,a is the immediate reward plus the discounted value of the max Q_a,s for the next state we would land in. 

Q-values are more convenient because the agent just have to make a decision based on the Q value of an action, nothing else. If it were using raw state values, it would need to know not just the values of the next states, but also the probabilities of transitions into those states. 

### Value iteration
* Initialize values for all states to be zero
* For every state, perform the Bellman update (V_s is equal to the max of the expected reward plus expected discounted value of the next state)
* Keep doing this over and over again

### Q-Value iteration
We can do the same thing with Q-values with only minor alterations:
* Initialize Q-values for all state/actions to be zero
* For every state state/action, perform the Bellman update (Q_s,a is equal to the max of the expected reward plus expected discounted max Q of the next state)
* Keep doing this over and over again


What are some potential problems with this? Well, our state space needs to be discrete and small enough to be able to do multiple iterations over every state. If it is not discrete, then maybe we need to discretize it, but then we are faced with more problems about how to do this. Secondly, we have to keep track of all the data about which action led from which state to which state (i.e. a transitions table), then use this to estimate probabilities. 











