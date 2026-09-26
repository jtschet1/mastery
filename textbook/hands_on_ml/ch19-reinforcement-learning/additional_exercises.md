# Chapter 19 — Reinforcement Learning

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Walk through the Gymnasium API for running an episode: what do gym.make(), env.reset(), and env.step() take and return?

gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=1000) creates an environment: render_mode="rgb_array" makes env.render() return frames as NumPy arrays, and max_episode_steps overrides the default episode length limit. env.reset(seed=42) starts an episode and returns (obs, info), where info is an environment-specific dict. env.action_space describes the valid actions (Discrete(2) means 0 or 1). env.step(action) returns five values: the new observation, the reward, done (the episode is over, e.g., the pole fell; Gymnasium's docs call it terminated), truncated (cut short, typically by the step limit), and info:

```python
obs, info = env.reset(seed=42)
total_rewards = 0
while True:
    obs, reward, done, truncated, info = env.step(policy(obs))
    total_rewards += reward
    if done or truncated:
        break
env.close()   # once you're completely done with the environment
```

After an episode ends, you must call reset() before stepping again.

## 2. How does a PyTorch policy network pick a discrete action, and why sample the action instead of taking the most likely one?

The network maps a state to action logits (for two actions, a single logit for action 1 is enough), with no final sigmoid or softmax, for performance and numerical stability. You wrap the logits in a probability distribution, sample from it, and keep the log probability, which the policy gradient loss needs:

```python
def choose_action(model, obs):
    logit = model(torch.as_tensor(obs))
    dist = torch.distributions.Bernoulli(logits=logit)  # Categorical if >2 actions
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)
```

Sampling balances exploration and exploitation: actions that worked become more likely, but the others still get tried now and then, so the agent can discover better ones. For continuous actions, the network instead outputs the mean and the (log) standard deviation of a Gaussian distribution to sample from.

## 3. Explain the REINFORCE algorithm and its loss function.

REINFORCE (Monte Carlo policy gradient) repeats these steps:

1. Play a full episode with the stochastic policy, recording each action's log probability and each reward.
2. Compute each action's return rₜ: the discounted sum of the rewards from that step on (reward + γ·next reward + γ²·the one after + …), computed backward through the episode.
3. Standardize the returns (subtract their mean, divide by their standard deviation) to stabilize training.
4. Take a gradient step on L(θ) = −∑ₜ log π_θ(aₜ | sₜ)·rₜ.

Minimizing L raises the probability of actions whose return was above average and lowers it for the others, which amounts to following the gradient of the expected return. It's simple but sample-inefficient, since single-episode returns are very noisy, and unstable, since it only trains on the states the current policy reaches, so it can forget how to handle the others.

## 4. State the Bellman optimality equation and explain how Q-value iteration turns it into an algorithm.

For a Markov decision process with known dynamics, the optimal state values satisfy, for every state s:

V*(s) = maxₐ ∑ₛ′ T(s, a, s′)·[R(s, a, s′) + γ·V*(s′)]

T(s, a, s′) is the probability of landing in s′ after taking action a in s, R(s, a, s′) is the reward for that transition, and γ is the discount factor. In words: a state's optimal value is what you get on average by taking the best action, the immediate reward plus the discounted optimal value of wherever you land.

Q-value iteration applies the same recursion to state-action pairs. Initialize all Q-values to 0 (−∞ for impossible actions) and repeatedly update every pair:

Qₖ₊₁(s, a) ← ∑ₛ′ T(s, a, s′)·[R(s, a, s′) + γ·maxₐ′ Qₖ(s′, a′)]

This dynamic-programming update converges to the optimal Q-values Q*(s, a), and the optimal policy is then simply π*(s) = argmaxₐ Q*(s, a).

## 5. What is the Q-learning update rule, and what are the TD target and the TD error?

Q-learning adapts Q-value iteration to the realistic case where the transition probabilities and rewards are unknown. Instead of averaging over all possible next states, it learns from each transition (s, a, r, s′) the agent actually experiences, keeping a running average:

Q(s, a) ← (1 − α)·Q(s, a) + α·(r + γ·maxₐ′ Q(s′, a′))

r + γ·maxₐ′ Q(s′, a′) is the TD target: the observed reward plus the value of acting optimally from s′ on, discounted by γ. Its difference from the current estimate Q(s, a) is the TD error, and each update moves Q(s, a) a fraction α of the way toward the target. TD learning does the same for state values: V(s) ← V(s) + α·(r + γ·V(s′) − V(s)). As with SGD, α must be gradually decreased for the estimates to settle instead of bouncing around. Once the Q-values are accurate, the policy is to pick argmaxₐ Q(s, a).

## 6. How does an ε-greedy exploration policy work, and what is an exploration function?

Q-learning only works if the exploration policy visits the environment thoroughly; a purely random policy eventually does, but it can take extremely long. An ε-greedy policy acts randomly with probability ε and greedily (highest estimated Q-value) with probability 1 − ε. As the estimates improve, it spends more and more time in the interesting parts of the environment while still visiting unknown regions. ε usually starts high (e.g., 1.0) and is gradually reduced (e.g., to 0.05).

An exploration function instead makes rarely tried actions look attractive by adding a bonus to the Q-values in the update target:

Q(s, a) ← (1 − α)·Q(s, a) + α·(r + γ·maxₐ′ f(Q(s′, a′), N(s′, a′)))

N(s′, a′) counts how often action a′ was chosen in state s′, and f(Q, N) = Q + κ/(1 + N), where the curiosity hyperparameter κ sets how strongly the agent is drawn to the unknown.

## 7. Walk through one deep Q-learning training step, including how the target Q-values are computed.

The DQN takes a state and outputs one Q-value per action, which is much more efficient than feeding it (state, action) pairs. Each training step works on a random batch of experiences sampled from the replay buffer:

```python
state, action, reward, next_state, done, truncated = sample_experiences(buffer, 32)
with torch.inference_mode():
    max_next_Q = model(next_state).max(dim=1).values
running = (~(done | truncated)).float()            # 0 where the episode ended
target_Q = reward + running * gamma * max_next_Q   # y = r + γ·maxₐ′ Q(s′, a′)
Q = model(state).gather(dim=1, index=action.unsqueeze(1))  # Q of the action taken
loss = criterion(Q, target_Q.unsqueeze(1))         # MSE or Huber
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The target is computed without tracking gradients, and it gets no future value if the episode ended. The chapter treats a truncated episode (cut short by a step limit) the same way for simplicity; some algorithms still add the future value there, since the state isn't truly terminal. gather() picks, in each row, the predicted Q-value of the action actually taken, and gradient descent pulls it toward the target. Because the agent needs the action with the highest Q-value at every step, DQNs generally don't suit continuous action spaces.

## 8. Why is basic deep Q-learning unstable, and how does a target network fix this?

The same network predicts Q(s, a) and also computes its own targets r + γ·maxₐ′ Q(s′, a′), so every update shifts the targets too, like a dog chasing its tail; training can oscillate, diverge, or freeze. The fix is to use two DQNs. The online model learns at every step and drives the agent, while the target model, a clone of it, is used only to compute the targets. The target model's weights are copied from the online model at regular intervals (e.g., every 10,000 steps for Atari), so the targets stay fixed in between and the feedback loop is damped:

```python
import copy
target_model = copy.deepcopy(model)                 # created once
# in each training step, compute the targets with the target model:
with torch.inference_mode():
    max_next_Q = target_model(next_state).max(dim=1).values
# every sync_interval steps:
target_model.load_state_dict(model.state_dict())
```

## 9. What problem does each of these DQN improvements address: double DQN, prioritized experience replay, and dueling DQN?

- Double DQN: taking the max over noisy Q-estimates tends to overestimate Q-values. So the online model picks the best next action and the target model evaluates it: y = r + γ·Q_target(s′, argmaxₐ′ Q_online(s′, a′)).
- Prioritized experience replay (PER): uniform sampling wastes time on unsurprising experiences. Each one gets priority p = |δ|, its latest TD error plus a small constant (new ones start very high so they're sampled at least once), and is sampled with probability P ∝ p^ζ (ζ = 0 means uniform). To compensate for this bias, its loss is weighted by w = (n·P)^(−β), where n is the buffer size and β is raised toward 1 during training.
- Dueling DQN: the network estimates a state value V(s) and an advantage A(s, a) per action, and outputs Q(s, a) = V(s) + A(s, a) − maxₐ′ A(s, a′), since the best action's advantage should be 0.

## 10. How does an actor-critic agent learn? Walk through its training step.

An actor-critic trains a policy network (the actor) and a value network (the critic) together. They often share their lower layers, with an actor head that outputs action logits and a critic head that estimates the state value V(s). At every environment step, the agent samples an action, observes r and s′, computes the target y = r + γ·V(s′) (just r if the episode ended) without tracking gradients, then:

```python
td_error = target_value - state_value               # δ = y − V(s)
actor_loss = -log_prob * td_error.detach()
critic_loss = criterion(state_value, target_value)  # e.g., MSE
loss = actor_loss + critic_weight * critic_loss     # critic weighted lower, e.g., 0.3
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The actor loss is REINFORCE's with the TD error in place of the return: it makes actions that did better than the critic expected more likely. detach() stops the actor loss from updating the critic. Using the critic's estimate instead of a full-episode return reduces the noise and lets the actor learn at every step, and like policy gradients, it supports stochastic policies and continuous actions.

## 11. What do A3C, A2C, SAC, and PPO each add to the basic actor-critic, and which would you reach for first?

- A3C (asynchronous advantage actor-critic): several agents explore their own copies of the environment in parallel, asynchronously pushing weight updates to a master network and pulling its latest weights; the critic estimates action advantages rather than state values.
- A2C: A3C without the asynchrony; synchronous updates over larger batches make better use of a GPU.
- SAC (soft actor-critic): maximizes the entropy of its actions as well as the rewards (be as unpredictable as possible while still earning rewards), which drives exploration and makes it very sample-efficient.
- PPO (proximal policy optimization): built on A2C, it clips the loss to prevent excessively large policy updates, which often destabilize training; it's a simpler take on TRPO.

Rule of thumb: PPO is a strong general-purpose default, SAC is the most sample-efficient for continuous actions (e.g., robotics), and DQN remains strong for discrete tasks like Atari or board games.

## 12. How do you train and use a PPO agent on an Atari game with Stable-Baselines3?

make_atari_env("BreakoutNoFrameskip-v4", n_envs=4) creates a vectorized environment bundling 4 copies of the game, stepped together with one action per copy, and preprocesses frames to 84 × 84 grayscale. Wrapping it in VecFrameStack(envs, n_stack=4) stacks the last 4 frames along the channel axis, so a single observation captures motion. Beware that SB3's vectorized API differs from Gymnasium's: reset() returns only the observations, and step() returns (obs, rewards, dones, infos), with no truncated flag.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
ppo = PPO("CnnPolicy", envs_stacked, n_steps=256, batch_size=256, n_epochs=4,
          clip_range=0.1, vf_coef=0.5, ent_coef=0.01, gamma=0.99)
ppo.learn(total_timesteps=10_000_000,
          callback=CheckpointCallback(save_freq=100_000, save_path="ckpts"))
ppo.save("ppo_breakout")                    # later: PPO.load("ppo_breakout")
action, _ = ppo.predict(obs, deterministic=True)
```

"CnnPolicy" builds a CNN with an actor head and a critic head. n_steps is the rollout length per environment before each update, n_epochs the number of optimization passes over that rollout, clip_range limits how much the policy can change per update, vf_coef weights the value loss, and ent_coef weights an entropy bonus that encourages exploration.
