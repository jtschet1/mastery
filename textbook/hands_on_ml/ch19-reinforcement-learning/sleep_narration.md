# Chapter 19 — Reinforcement Learning, to fall asleep to

Chapter nineteen. Reinforcement learning.

---

Deep reinforcement learning.

Reinforcement learning, or RL, is one of the oldest fields of machine learning. It has existed since the 1950s, with applications in games, such as TD-Gammon, a backgammon program, and in machine control, but it rarely made headlines.

That changed in 2013, when researchers at DeepMind, a British startup, showed a system that learned to play almost any Atari game from scratch, using only raw pixels as input and no prior knowledge of the rules, and eventually outperformed humans in most of them. Applying deep learning to reinforcement learning created deep reinforcement learning.

More results followed. DeepMind's AlphaGo beat the legendary Go player Lee Sedol in 2016 and the world champion Ke Jie in 2017. AlphaFold, in 2020, predicts the three-dimensional shape of proteins with unprecedented accuracy, and earned Demis Hassabis and John Jumper the Nobel Prize in Chemistry. AlphaCode, in 2022, generates code at a competitive programming level. GNoME, in 2023, predicted hundreds of thousands of new stable crystal structures.

The chapter covers three families of techniques: policy gradients; deep Q-networks, which build on Markov decision processes; and actor-critic methods, including proximal policy optimization, or PPO.

---

What reinforcement learning is.

In reinforcement learning, a software agent makes observations and takes actions within an environment, and receives rewards in return. Its objective is to learn to act in a way that maximizes its expected rewards over time. Positive rewards can be thought of as pleasure and negative rewards as pain, and the agent learns by trial and error.

The agent can be a program controlling a robot, observing the real world through sensors such as cameras, sending signals to motors, and rewarded for approaching its target. It can control Ms. Pac-Man, where the actions are the nine joystick positions, the observations are screenshots, and the rewards are the game points. It can play Go and only get a reward when it wins. It can be a smart thermostat, rewarded for staying near the target temperature and saving energy, and penalized when humans have to adjust it. It can trade on the stock market, with gains and losses as rewards.

There may be no positive rewards at all: an agent in a maze might get a negative reward at every step, so it must find the exit fast. Other applications include self-driving cars, recommender systems, ad placement, and controlling where an image classifier focuses its attention.

---

Policies and policy search.

The algorithm an agent uses to determine its actions is its policy. It can be any algorithm, for example a neural network that takes observations as inputs and outputs the action to take.

A policy does not have to be deterministic, or even observe the environment. A blind robotic vacuum cleaner, rewarded for the dust it collects in thirty minutes, could move forward with probability p each second, or otherwise rotate by a random angle between minus r and plus r. A policy that involves randomness is called a stochastic policy. The resulting erratic trajectory guarantees that the robot eventually reaches every place it can reach.

Training this robot means tuning two policy parameters, p and r. Trying many combinations and keeping the best is policy search by brute force, which is hopeless when the policy space is large, as it usually is.

Genetic algorithms are another way to explore the policy space. For example, one hundred random policies are tried, the eighty worst are killed, and the twenty survivors each produce four offspring, which are copies of their parent plus some random variation. Survivors and offspring form the next generation, and the process repeats until a good policy emerges. Giving poor performers a slight chance of survival often helps preserve diversity.

A third approach evaluates the gradients of the rewards with regard to the policy parameters, and tweaks the parameters by following the gradients toward higher rewards. This is gradient ascent, and algorithms that use it are called policy gradient algorithms, or PG algorithms.

---

The Gymnasium library.

Training an agent requires an environment. Training in the real world is hard and slow: there is no undo if it falls off a cliff, time cannot be sped up, and running a thousand robots in parallel is too expensive. So simulated environments are generally used, at least for bootstrap training, for example with PyBullet or MuJoCo for three-dimensional physics.

Gymnasium is an open source toolkit, the successor of OpenAI Gym, that provides many simulated environments, including Atari games, board games, and two- and three-dimensional physics simulations. The Arcade Learning Environment, or ALE, emulates the Atari 2600 for the Atari environments, and the Box2D library provides two-dimensional physics for several others.

The chapter uses CartPole: a two-dimensional simulation in which a cart is accelerated left or right to balance a pole on top of it. The environment's reset method starts an episode and returns the first observation plus a dictionary of extra information, which in many Atari environments holds the number of lives left, and in CartPole is empty. The render method returns the current frame as an image array. A CartPole observation holds four floats: the cart's horizontal position, its velocity, the pole's angle, where zero is vertical, and its angular velocity. There are two discrete actions: zero accelerates left, and one accelerates right.

The step method executes an action and returns five values: the new observation; the reward, which in CartPole is one at every step, so the goal is to keep the episode going as long as possible; a done flag, true when the episode is over because the pole tilted too much or went off screen; a truncated flag, true when the episode was cut short, typically by a maximum number of steps, here one thousand; and the info dictionary. An episode is one run of the environment until the game is over or interrupted. Some algorithms treat truncated episodes differently from finished ones, but the chapter treats them the same.

A hardcoded policy that accelerates in the direction the pole leans averages about forty-two steps over five hundred episodes, and never lasts more than sixty-three. The cart oscillates more and more until the pole falls.

---

Neural network policies.

A neural network policy takes an observation and estimates a probability for each action, then samples an action according to those probabilities. CartPole needs a single output: the probability p of action one, with action zero having probability one minus p. This is a Bernoulli distribution.

Sampling, rather than always taking the highest-scoring action, balances exploring new actions with exploiting actions known to work. At a new restaurant, a dish that turns out well should become more likely to be ordered again, but not certain, or better dishes will never be tried. This exploration versus exploitation dilemma is central to reinforcement learning.

In CartPole, past observations can be ignored, because each observation contains the environment's full state and is noise-free. If part of the state were hidden, such as the cart's velocity, or if observations were noisy, past observations would be needed to estimate the current state.

The PyTorch policy network is a tiny multilayer perceptron: four inputs, one hidden layer of five units with ReLU, and one output. With more actions there would be one output per action. For performance and numerical stability, it outputs a logit rather than a probability, with no sigmoid at the end. The action-choosing function builds a Bernoulli distribution from the logit, samples an action, and also returns the log probability of that action, which training needs. With more than two actions, a categorical distribution is used. For continuous actions, a Gaussian distribution is used, and the network predicts its mean and its standard deviation, or the log of the standard deviation, which is often clipped so the distribution is neither too wide nor too narrow.

---

The credit assignment problem.

If the best action at each step were known, the network could be trained by ordinary supervised learning. In reinforcement learning, the only guidance is rewards, which are typically sparse and delayed. If the pole falls after one hundred steps, the agent cannot tell which actions were good and which were bad; the last action is surely not solely to blame. This is the credit assignment problem: it is hard to know which actions deserve credit or blame for a reward. A dog rewarded hours after behaving well has the same problem.

A common strategy is to evaluate each action by the sum of all the rewards that follow it, discounted by a factor gamma at each step. This is the action's return. For example, with rewards of plus ten, zero, and minus fifty over three steps, and gamma equal to zero point eight, the first action's return is ten, plus zero point eight times zero, plus zero point eight squared times minus fifty, which is minus twenty-two. The three actions' returns are minus twenty-two, minus forty, and minus fifty, computed by walking backward through the rewards and adding to each reward the discounted return of the next step.

A discount factor near zero makes future rewards count for little; near one, distant rewards count almost as much as immediate ones. Typical values range from zero point nine to zero point nine nine. With zero point nine five, rewards thirteen steps ahead count about half as much as immediate ones; with zero point nine nine, the halfway point is sixty-nine steps. CartPole's actions have short-term effects, so zero point nine five is reasonable; it helps credit assignment, making training faster and more stable. A discount factor that is too low makes the agent focus too much on short-term gains.

---

The REINFORCE algorithm.

REINFORCE, also called Monte Carlo policy gradients, was introduced in 1992 by Ronald Williams. The policy plays a full episode while the rewards and the log probabilities of its actions are recorded. Each action's return is computed. Then actions with positive returns are made more likely and actions with negative returns less likely, by minimizing the REINFORCE loss.

The REINFORCE loss is the negative of the sum, over all time steps, of the log probability the policy gave to the action it took, multiplied by that action's return. Minimizing it maximizes the expected discounted rewards.

The training loop runs an episode, computes the returns, standardizes them by subtracting their mean and dividing by their standard deviation, computes the loss, and takes an optimizer step. Standardizing is optional but recommended, because it stabilizes training. With the NAdam optimizer, a learning rate of zero point zero six, a discount factor of zero point nine five, and two hundred episodes, training takes under a minute, and the policy balances the pole perfectly.

REINFORCE does not scale well to complex tasks. It is highly sample inefficient, because its return estimates are very noisy, especially when good actions are mixed with bad ones. It is also unstable: the agent may improve, forget everything, and learn again. This is largely because the training samples are not independent and identically distributed, or IID; they are whatever states the agent currently reaches. Once it learns to keep the pole upright, it no longer sees tilted poles, and forgets how to handle them. Still, REINFORCE is the foundation of more powerful methods, such as actor-critics.

Outside of research, injecting prior knowledge speeds up training dramatically. For example, negative rewards proportional to the pole's angle make rewards much less sparse, and a network can first be trained to imitate a reasonable hardcoded policy.

Reinforcement learning is notoriously difficult, because of training instabilities and huge sensitivity to hyperparameters and random seeds. Andrej Karpathy put it this way: supervised learning wants to work, while RL must be forced to work. This is a major reason RL is less widely adopted than regular deep learning.

---

Markov decision processes.

Value-based methods are less direct than policy gradients. The agent learns to estimate the value of each state, meaning its expected return, or the value of each action in each state, and uses those estimates to decide how to act. They rest on Markov decision processes, or MDPs.

The mathematician Andrey Markov studied stochastic processes with no memory, called Markov chains. A Markov chain has a fixed number of states and moves randomly between them at each step. The probability of moving from a state s to a state s prime depends only on that pair of states, not on past states. In the book's example, a process starting in state s zero stays there with seventy percent probability at each step, eventually leaves and never returns, bounces between s one and s two for a while, and finally falls into s three, which it can never leave. A state with no way out is a terminal state. Markov chains are used in thermodynamics, chemistry, statistics, and many other fields.

Richard Bellman first described Markov decision processes in the 1950s. At each step, an agent chooses among several actions, and the transition probabilities depend on the chosen action. Some transitions also give a reward, and the agent's goal is a policy that maximizes its cumulative reward over time.

The book's example MDP has three states. In s zero, action a zero gives plus ten with seventy percent probability and keeps the agent in s zero, but sooner or later moves it to s one. In s one, action a zero stays put, and action a two leads to s two at a cost of minus fifty. In s two, the only action leads back to s zero most of the time, with a reward of plus forty. The open question is whether, in s one, the agent should stay put or go through the fire.

---

The Bellman optimality equation and value iteration.

The optimal state value of a state s, written V star of s, is the sum of discounted future rewards the agent can expect on average starting from s, if it acts optimally. The Bellman optimality equation says that this value equals the maximum, over all actions, of the expected result of taking that action. That expected result is a sum over all possible next states: the transition probability to that next state, times the reward for the transition plus gamma times the optimal value of the next state. In short, the optimal value of a state is the average reward of one optimal action plus the discounted expected optimal value of where it leads. In the example MDP, taking the only action in s two leads to s zero with probability zero point eight and a reward of plus forty.

An optimal policy here maximizes discounted rewards, so it depends on gamma. Real tasks usually care about undiscounted rewards per episode, which is also how agents are evaluated, so gamma is set close to one, but not so close that training becomes slow and unstable.

The value iteration algorithm turns the equation into an update. All estimates start at zero, and each iteration replaces every state's estimate with the right-hand side of the Bellman equation, computed from the previous estimates. Given enough time, the estimates are guaranteed to converge to the optimal state values. This is an example of dynamic programming, which breaks a complex problem into tractable subproblems tackled iteratively.

---

Q-value iteration.

Optimal state values help evaluate a policy, but do not directly give the optimal policy. Bellman found a similar algorithm for optimal state-action values, called Q-values, for quality values. The optimal Q-value of a state and an action, written Q star of s and a, is the sum of discounted future rewards the agent can expect on average after choosing action a in state s, before seeing the outcome, assuming it acts optimally afterward.

Q-value iteration starts all estimates at zero and repeatedly updates them. The new Q-value of a state and action is the sum, over next states, of the transition probability times the reward plus gamma times the highest previous Q-value available in the next state.

Once the optimal Q-values are known, the optimal policy, pi star, is simple: in each state, choose the action with the highest Q-value, the argmax.

On the example MDP, impossible actions get Q-values of minus infinity, and fifty iterations with gamma equal to zero point nine are enough. For example, choosing a one in s zero turns out to be worth about seventeen. The optimal policy is to choose a zero in s zero, stay put in s one, and take the only action in s two. With gamma equal to zero point nine five, the best action in s one becomes going through the fire. The more future rewards are valued, the more short-term pain is worth accepting.

---

Temporal difference learning.

RL problems with discrete actions can often be modeled as MDPs, but the agent initially knows neither the transition probabilities nor the rewards. It must experience each transition at least once to know its reward, and many times to estimate its probability.

Temporal difference learning, or TD learning, adapts value iteration to this situation. The agent knows only the possible states and actions. It explores with an exploration policy, for example a random one, and updates its value estimates from the transitions and rewards it actually observes.

The TD target is the observed reward plus gamma times the estimated value of the next state. The TD error is the TD target minus the current estimate. Each update moves the current estimate a fraction alpha, the learning rate, of the way toward the target: the new estimate is one minus alpha times the old estimate, plus alpha times the TD target. Equivalently, the estimate is increased by alpha times the TD error. So TD learning keeps a running average of the immediate reward plus the discounted value expected afterward.

TD learning resembles stochastic gradient descent: it handles one sample at a time, and it only truly converges if the learning rate is gradually reduced.

---

Q-learning.

Q-learning similarly adapts Q-value iteration to unknown transition probabilities and rewards. It watches an agent play, for example randomly, and gradually improves its Q-value estimates. The update moves the Q-value of a state and action toward the observed reward plus gamma times the highest Q-value estimate in the next state. The maximum is used because the target policy is assumed to act optimally from then on. Once the estimates are accurate, the optimal policy is the greedy policy: always choose the action with the highest Q-value.

On the example MDP, the exploration policy is purely random, which is enough for such a small state space. The learning rate decays by power scheduling: an initial rate of zero point zero five divided by one plus the iteration number times zero point zero zero five. Q-learning converges to the optimal Q-values, but it needs about eight thousand iterations, while Q-value iteration needs fewer than twenty. Not knowing the transition probabilities or rewards makes the problem much harder.

Q-learning is off-policy: the policy being trained is not the one being executed. The agent acted completely randomly, while the greedy policy being learned was never used. REINFORCE, by contrast, is on-policy: it explores using the policy being trained. It is somewhat surprising that Q-learning learns an optimal policy by watching random play, like learning golf from a blindfolded monkey.

---

Exploration policies.

Q-learning works only if the exploration policy explores the MDP thoroughly. A purely random policy eventually visits everything, but it can take an extremely long time.

A better option is the epsilon-greedy policy: at each step, act randomly with probability epsilon, or greedily with probability one minus epsilon. As the Q-value estimates improve, it spends more time in the interesting parts of the environment while still visiting unknown regions. Epsilon commonly starts at one and is gradually reduced, for example to zero point zero five.

Another approach rewards trying actions that have rarely been tried, with a bonus added to the Q-value estimates. One such exploration function is the Q-value plus kappa divided by one plus the number of times that action was chosen in that state. Kappa is a curiosity hyperparameter: how strongly the agent is drawn to the unknown.

---

Approximate Q-learning and deep Q-learning.

Q-learning does not scale to large MDPs. Ms. Pac-Man has about two hundred forty pellets, each present or eaten, giving about two to the power two hundred forty pellet states, roughly ten to the power seventy-three. Adding the positions of the ghosts and Ms. Pac-Man gives more states than atoms in the galaxy, far too many to track one Q-value each.

The solution is a function with parameters theta that approximates the Q-value of any state-action pair. This is approximate Q-learning. For years it relied on linear combinations of handcrafted features, such as the distances to the nearest ghosts. In 2013, DeepMind showed that deep neural networks work much better, especially on complex problems, with no feature engineering. A deep neural network that estimates Q-values is a deep Q-network, or DQN, and using one for approximate Q-learning is deep Q-learning.

The training target comes from Bellman. The target Q-value for a state and action is the observed reward plus gamma times the highest Q-value the DQN predicts for the next state. Training minimizes the squared error between the DQN's estimate and this target, or the Huber loss, which is less sensitive to large errors.

---

Implementing deep Q-learning.

In practice, a DQN takes only the state as input and outputs one Q-value per possible action, which is much more efficient than taking a state-action pair. For CartPole it has two hidden layers of thirty-two units. Actions are chosen with an epsilon-greedy policy.

DQNs generally do not work with continuous action spaces, because finding the action with the highest Q-value would require running an optimization at every step. The exceptions are a space tiny enough to discretize, or a DQN combined with policy gradients.

All experiences are stored in a replay buffer, also called replay memory, and each training step samples a random batch from it. This reduces correlations within a batch and makes the data distribution more consistent, which stabilizes training. Each experience records the state, the action, the reward, the next state, and whether the episode was done or truncated. The buffer here is a double-ended queue from Python's collections package with a maximum length, so each new experience replaces the oldest. Random access in a very long deque is slow, so very large buffers should use a circular buffer or DeepMind's Reverb library.

A training step samples a batch and computes the targets in PyTorch's inference mode: the DQN predicts the Q-values of each next state, the maximum is multiplied by the discount factor, zeroed out if the episode had ended, and added to the reward. The DQN then predicts the Q-values of the current states, the Q-value of each action actually taken is extracted, and the mean squared error, or MSE, between these and the targets is minimized. Inference mode is like disabling gradient tracking, except that models also run in evaluation mode and the resulting tensors cannot be used in backpropagation.

The full loop runs eight hundred episodes. Each episode is played and recorded, then one training step runs, starting after thirty warmup episodes so the buffer holds enough experiences. Epsilon decreases linearly from one to zero point zero one over five hundred episodes, so the agent shifts from exploration to exploitation. The run uses batches of thirty-two experiences, a discount factor of zero point nine five, and a buffer of up to one hundred thousand experiences.

The total reward per episode is a better progress indicator than the loss. The loss can fall while the agent gets worse, for example when it is stuck in one small region and the DQN overfits it, and it can rise while the agent improves, for example when the DQN corrects underestimated targets.

The agent eventually balances the pole perfectly, reaching the maximum reward of one thousand, but training is completely unstable, even less stable than REINFORCE. It reached two hundred points after about two hundred episodes, forgot everything, and only cracked the problem around episode five hundred fifty. The cause is that the model sets its own targets, like a dog chasing its own tail. That feedback loop can make the network diverge, oscillate, or freeze.

---

DQN improvements.

DeepMind's 2013 paper stabilized training with two DQNs. The online model learns at each step and moves the agent. The target model, a clone of the online model, is used only to compute targets, and its weights are copied from the online model at regular intervals, for example every ten thousand steps. Stable targets damp the feedback loop. The paper also used a very large replay buffer, a tiny learning rate, fifty million training steps, an epsilon that decreased slowly over one million steps, and a convolutional neural network, or CNN.

Double DQN, from 2015, addresses the target network's tendency to overestimate Q-values. If all actions are equally good, their estimates differ only by noise, and taking the maximum picks one that is likely above the true value, like measuring a pool's depth by its tallest random wave. The fix is to select the best next action with the online model, and estimate its Q-value with the target model.

Prioritized experience replay, or PER, also from 2015, samples important experiences more often. An experience is important if it is likely to lead to fast learning, which can reasonably be measured by the magnitude of its TD error: a large error means a surprising transition. New experiences get a very high priority, so they are sampled at least once. After each sampling, the priority is set to the absolute TD error plus a small constant. The sampling probability is proportional to the priority raised to the power zeta. A zeta of zero gives uniform sampling and one gives full importance sampling; the paper used zero point six.

Because important experiences are oversampled, their training weights must be reduced, or the model overfits them. Each experience's weight is the number of experiences in the buffer times its sampling probability, raised to the power minus beta. A beta of zero means no compensation and one means full compensation. The paper increased beta from zero point four to one over training. Raising zeta usually calls for raising beta too.

Dueling DQN, also from 2015, is distinct from double DQN, though the two combine easily. A Q-value can be split into the value of the state plus the advantage of the action over the other actions. The best action's Q-value equals the state's value, so its advantage is zero. A dueling DQN estimates the state value and each action's advantage separately, and subtracts the maximum predicted advantage from all advantages so the best one is zero.

DeepMind's 2017 Rainbow agent combined six different techniques and largely outperformed the state of the art.

---

Actor-critic algorithms.

Actor-critics combine policy gradients with value-based methods. The actor is a policy and the critic is a value network, and both are trained together. The actor relies on the critic's estimates of state values or action advantages to guide its updates, like an athlete working with a coach. Since the critic can use a large replay buffer, training is more stable and data efficient. Like policy gradients, actor-critics support stochastic policies and continuous action spaces.

The CartPole actor-critic has a shared body of two hidden layers, an actor head that outputs the action logit, and a critic head that outputs the state value. Sharing the body reduces the number of parameters and improves data efficiency, but couples the two networks more tightly, which makes training a bit less stable.

The actor-critic trains at every step. The target value is the reward plus gamma times the critic's value of the next state, or just the reward if the episode ended, and it is computed without gradients. The TD error is the target minus the critic's value of the current state. The actor's loss is minus the log probability of the action times the TD error, so actions that did better than the critic expected are encouraged. The TD error is detached here, so the actor's loss does not train the critic. The critic's loss is the MSE between its value and the target. The total loss is the actor's loss plus a smaller weight, here zero point three, times the critic's loss, which helps stability.

After four hundred episodes, the agent balances the pole and collects the maximum reward, but the method remains very sensitive to hyperparameters and random seeds.

---

Actor-critic variants.

Asynchronous advantage actor-critic, or A3C, introduced by DeepMind in 2016, has multiple agents learning in parallel on separate copies of the environment. Each one periodically and asynchronously pushes weight updates to a master network and pulls the latest weights. Its critic estimates the advantage of each action rather than state values, as in dueling DQN, which is where the second A in the name comes from.

Advantage actor-critic, or A2C, removes the asynchronicity. Updates are synchronous and use larger batches, which makes better use of the GPU.

Soft actor-critic, or SAC, proposed in 2018 by Tuomas Haarnoja and other UC Berkeley researchers, maximizes both rewards and the entropy of its actions: it tries to be as unpredictable as possible while still collecting rewards. This encourages exploration, speeds up training, and makes the agent less likely to keep repeating the same action when the critic's estimates are imperfect. It is remarkably sample efficient.

Proximal policy optimization, or PPO, by John Schulman and other OpenAI researchers, is based on A2C but clips the loss function to avoid excessively large weight updates, which often cause instability. It simplifies the earlier trust region policy optimization algorithm, or TRPO, also from OpenAI. In April 2019, OpenAI Five, based on PPO, beat the world champions at Dota 2.

SAC and PPO are among the most widely used RL algorithms today. PPO is a good general-purpose default. SAC is the most sample efficient for continuous actions, which suits robotics. DQN remains strong for discrete tasks such as Atari or board games.

---

PPO on Atari Breakout with Stable-Baselines3.

Stable-Baselines3, or SB3, is a library with optimized implementations of popular RL algorithms. The chapter uses its PPO to train an agent on the Atari game Breakout, with the ALE emulator and the freely available game files.

An SB3 helper creates a wrapper that runs four Breakout environments in parallel and converts each frame from a two hundred ten by one hundred sixty color image to an eighty-four by eighty-four grayscale image. The emulator runs at sixty frames per second, so consecutive frames are nearly identical. Rather than frame skipping, where each action is repeated four times and only the last frame is returned, the chapter uses the no-frameskip version of Breakout and a frame-stacking wrapper, which repeats each action four times and stacks the four frames into one four-channel observation.

The PPO model uses SB3's CNN policy, a deep convolutional network with an actor head for the action logits and a critic head for state values. Breakout has four actions: left, right, fire, and no-op. The key hyperparameters are the learning rate; the batch size; the number of steps each environment runs before each policy update; the number of epochs run on each batch; the clip range, which limits how much the policy can change in one update; the value function coefficient, which weights the critic's loss; the entropy coefficient, which weights the entropy term that encourages exploration; and gamma. The chapter uses a learning rate of two point five times ten to the minus four, a batch size and a step count of two hundred fifty-six each, four epochs, a clip range of zero point one, a value function coefficient of zero point five, an entropy coefficient of zero point zero one, and a gamma of zero point nine nine. A custom network requires subclassing SB3's actor-critic policy class.

The defaults are a good start for new tasks. If learning is too slow, more parallel environments help first, then a higher learning rate or a larger clip range. Smaller step counts or batches also speed things up, at the risk of noisier gradients. If learning is unstable, a lower learning rate or clip range and larger rollouts or batches help. A higher entropy coefficient encourages more exploration. Gamma should be about zero point nine five for short-horizon tasks, and zero point nine nine five to zero point nine nine nine for long-horizon ones.

Training runs for thirty million steps over many hours, so a checkpoint callback saves the model at regular intervals in case of a crash. TensorBoard shows the learning curves; the key metric is the mean reward per episode. After one million steps it is around twenty, which is poor; after ten million it reaches human level; after fifty million it is generally superhuman. For viewing, the trained agent plays an episode with deterministic predictions while the frames are recorded. A trained agent even finds one of the best strategies in the game: digging a tunnel along the side and sending the ball through it.

---

AlphaGo and model-based reinforcement learning.

AlphaGo uses a variant of Monte Carlo tree search, or MCTS, invented in 1949 by Nicholas Metropolis and Stanislaw Ulam. MCTS runs many simulations from the current position, spending more time on the most promising branches. When it reaches a new node, it plays randomly until the game ends, then updates the estimates of the visited nodes based on the outcome.

AlphaGo replaces the random play with a policy network trained with policy gradients. The original system used three additional networks. AlphaGo Zero simplified it to a single network that both selects moves and evaluates positions. AlphaZero extended it beyond Go to chess and shogi. MuZero outperformed them all, even though it starts without knowing the rules of the game.

AlphaGo had the rules of Go hardcoded, while MuZero learns a model of the environment, predicting the reward and next-state probabilities for each state and action. Having a model of the environment, hardcoded or learned, allows planning ahead, so these are model-based RL algorithms. Policy gradients, value-based methods, and actor-critics are model-free: they model a policy, values, or both, but not the environment.

---

Curiosity-based exploration.

Sparse rewards make RL slow. Deepak Pathak and other UC Berkeley researchers proposed ignoring the environment's rewards and making the agent curious instead, so its rewards are intrinsic. The agent constantly predicts the outcomes of its actions and seeks situations where its predictions fail. Predictable outcomes are boring, so it moves on, and unpredictable outcomes it cannot control also become boring over time. With curiosity alone, agents learned to play many video games. Losing is not penalized, but it is boring because the game starts over, so the agent learns to avoid it.

---

Open-ended learning.

Open-ended learning, or OEL, aims to train agents that endlessly learn new and interesting tasks, typically generated procedurally. Uber AI's 2019 POET algorithm generates many two-dimensional environments with bumps and holes and trains one walking agent per environment. The environments start simple and get harder, which is called curriculum learning. Agents regularly compete across all environments, and in each environment the winner replaces the current agent, so knowledge transfers between environments and the most adaptable agents survive. The resulting agents walk much better than agents trained on a single task and handle much harder terrain. Follow-up work includes the Enhanced POET paper and a 2021 DeepMind paper on open-ended learning.

---

Summary.

In reinforcement learning, an agent acts in an environment and learns a policy that maximizes its expected rewards over time. Stochastic policies balance exploration and exploitation.

Rewards are sparse and delayed, which causes the credit assignment problem. Actions are judged by their return, the discounted sum of the rewards that follow. The discount factor controls how much the future counts, and changing it can change the optimal policy.

Policy gradient methods, such as REINFORCE, make actions with high returns more likely. They are simple but sample inefficient and unstable.

Value-based methods build on Markov decision processes and the Bellman equation. Value iteration and Q-value iteration need a known model; TD learning and Q-learning learn from experience. Q-learning is off-policy and needs good exploration, such as epsilon-greedy. Deep Q-learning approximates Q-values with a neural network and a replay buffer, and is stabilized by a target network, double DQN, prioritized experience replay, and dueling DQN.

Actor-critics train a policy and a value network together. PPO is a strong default, SAC excels at continuous control, and DQN remains strong for discrete actions. Stable-Baselines3 provides ready-made implementations, and its PPO can learn to play Breakout at a superhuman level.

Model-based methods, such as AlphaGo and MuZero, plan with a model of the environment. Curiosity-based exploration replaces sparse rewards with intrinsic ones, and open-ended learning aims for agents that keep learning new tasks.
