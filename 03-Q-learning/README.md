# Q-Learning and the Bellman Equation

## Value iteration

We need 3 central data structures:

* Reward table

We can structure this anyway we want, but we'll do it as a dict with the key being a composite of (source_state, action, target_state). The value is then the immediate reward

* Transitions table

This will also be a dict, but one that keeps count of the transition probabilities for entering a target state when choosing an action in a source state. Key will be a composite (source_state, action) and the value will be the counts of the states the agent ends up in. 

* Value table

A dict that has a state as its key and its value as the value. 

The logic is thenas follows: play a number of random steps to populate the reward and transition tables. After the random steps we perform value iteration over all states in order to update our value table. Then we run some full episodes using the value table to see if we can 'solve' the environment. During these full episodes we continue to update the reward and transition table. 

### Value iteration notes:
We don't need to wait for the end of the episode to start learning; we just perform N steps and remember their outcomes. This is different from the cross-entropy method which could only learn on full episodes. 







