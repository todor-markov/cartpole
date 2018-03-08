# cartpole
Implements three agents for the OpenAI gym cartpole environment. All agents use a linear policy:
1. An agent that randomly draws new policy parameters from a pre-specified distribution at each iteration. If the new parameters perform worse than the old ones, they are discarded; if they perform better, they are retained.
2. A solver that adds gaussian noise to the current policy parameters at each iteration. If the new parameters perform worse than the old ones, they are discarded; if they perform better, they are retained.
3. An agent that updates its policy parameters using policy gradients. The policy gradients implementation is based on Sergey Levine's [slides][1] from Berkeley's [Deep Reinforcement Learning][2] course

To run an agent, type `python cartpole.py <agent_type>` with `<agent_type>` replaced by `rgo`, `hco` or `pgo` for the first, second and third agent respectively (These abbreviations stand for 'random guessing optimization', 'hill climbing optimization', and 'policy gradients optimization' respectively)

[1]: http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf
[2]: http://rll.berkeley.edu/deeprlcourse/