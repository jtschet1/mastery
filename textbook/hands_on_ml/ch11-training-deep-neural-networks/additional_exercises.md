# Chapter 11 — Training Deep Neural Networks

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. State the Glorot, He, and LeCun weight-initialization rules, and say which activation functions each one is meant for.

Each draws every weight independently from a zero-mean distribution whose variance depends on the layer's fan-in (number of inputs) and/or fan-out (number of outputs), chosen so that activations and gradients keep roughly the same variance from layer to layer:

- Glorot (Xavier): σ² = 1 / fan_avg, where fan_avg = (fan_in + fan_out) / 2. For no activation, tanh, sigmoid, or softmax.
- He (Kaiming): σ² = 2 / fan_in. For ReLU and its variants (leaky ReLU, ELU, GELU, Swish, Mish, ...); the factor 2 compensates for ReLU outputting zero for roughly half of its inputs.
- LeCun: σ² = 1 / fan_in, i.e., Glorot with fan_in in place of fan_avg. For SELU, preferably with a normal distribution.

To sample from a uniform distribution instead, use the range −r to +r with r = √(3σ²).

## 2. How would you apply He initialization to every nn.Linear layer of a PyTorch model, and why override the default initialization at all?

nn.Linear initializes its weights with Kaiming uniform scaled down by a factor of √6 (and its biases randomly), which isn't the right scale for any common activation function. The torch.nn.init module provides in-place initializers (hence the trailing underscore) such as kaiming_uniform_, kaiming_normal_, and zeros_. To initialize a whole model, write a function that handles one module and pass it to model.apply(), which calls it on every submodule recursively:

```python
def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1))
model.apply(use_he_init)
```

For leaky ReLU with slope α, also pass a=α and the nonlinearity, e.g. nn.init.kaiming_uniform_(w, a=0.2, nonlinearity="leaky_relu"), which divides the variance by 1 + α².

## 3. Walk through what a batch-normalization layer computes during training, and what changes at inference time.

During training, BN standardizes each input feature using statistics of the current mini-batch B of m_B instances, then rescales and shifts it:

1. μ_B = (1/m_B) ∑ᵢ xᵢ
2. σ_B² = (1/m_B) ∑ᵢ (xᵢ − μ_B)²
3. x̂ᵢ = (xᵢ − μ_B) / √(σ_B² + ε)
4. zᵢ = γ ⊗ x̂ᵢ + β

All of this is per feature: ε is a tiny smoothing term that avoids division by zero, and the scale γ and offset β are learned by backprop, letting the network pick the best scale and mean for each input.

At inference you may have a single instance, or a small or non-independent batch, so batch statistics would be unavailable or unreliable. Instead, during training BN also keeps exponential moving averages of the batch means and variances, and in evaluation mode it uses these running μ and σ² in place of μ_B and σ_B².

## 4. What PyTorch details matter when you add nn.BatchNorm1d or nn.BatchNorm2d layers to a model?

- The constructor takes the number of features (channels, for BatchNorm2d). BatchNorm1d expects inputs of shape [batch, features]; BatchNorm2d expects [batch, channels, height, width] and computes its statistics over the batch and both spatial dimensions, so it learns one γ and one β per channel.
- γ and β are the layer's weight and bias parameters. The running mean and variance are buffers (running_mean, running_var), updated during training but not by the optimizer.
- Call model.train() before training and model.eval() before evaluating or predicting: forgetting to switch is one of the most common bugs.
- When BN comes right after a linear or convolutional layer (before the activation), create that layer with bias=False, since BN's β already provides an offset.
- momentum is the weight given to the new batch statistic in the running average (default 0.1), the reverse of the usual convention; values such as 0.01 work better for small batches.

## 5. How does layer normalization differ from batch normalization, and how do you tell nn.LayerNorm which dimensions to normalize over?

Batch norm normalizes each feature using statistics computed across the instances in the batch; layer norm normalizes each instance using statistics computed across its own features. So LN behaves identically during training and inference, needs no running averages, works with any batch size, and an instance's output never depends on the other instances in its batch. That's why it's popular in recurrent nets and standard in transformers, and it's increasingly used in CNNs. Like BN, it learns a scale and an offset for each normalized element.

nn.LayerNorm takes the shape of the trailing dimension(s) to normalize over. For images batched as [32, 3, 100, 200]:

- nn.LayerNorm([100, 200]) normalizes each channel of each image separately.
- nn.LayerNorm([3, 100, 200]) normalizes over all channels at once, which is what most vision architectures that use LN do.

## 6. How do you add gradient clipping to a PyTorch training loop, and how do clip_grad_norm_ and clip_grad_value_ differ?

Clip after computing the gradients and before the optimizer uses them:

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

- clip_grad_norm_ rescales the gradients whenever their ℓ2 norm (all the given parameters' gradients taken together as one vector) exceeds max_norm, so the update keeps its direction.
- clip_grad_value_(model.parameters(), clip_value=1.0) clamps each gradient component to between −clip_value and +clip_value independently, which can change the direction: [0.9, 100.0] becomes [0.9, 1.0], whereas clipping by norm gives about [0.009, 1.0].

Both work well in practice; which is better depends on the dataset. Clipping is mainly used against exploding gradients in recurrent nets, where batch norm is hard to use.

## 7. Walk through reusing the layers of a pretrained PyTorch model, model_A, to build a binary classifier for a similar task.

Deep-copy every layer except model_A's output layer, add a new head, and freeze the reused layers at first:

```python
import copy
reused_layers = copy.deepcopy(model_A[:-1])  # all layers but the old head
new_head = nn.Linear(100, 1)  # 100 features in, 1 logit out
model_B = nn.Sequential(*reused_layers, new_head).to(device)
for layer in model_B[:-1]:
    for param in layer.parameters():
        param.requires_grad = False  # gradient descent won't change them
```

Without the deep copy, model_B would share its layers with model_A, so training B would modify A too. Freezing protects the pretrained weights from the large error gradients that the randomly initialized head produces early on. Train with a loss suited to the new task (here nn.BCEWithLogitsLoss, since the head outputs one logit) for a few epochs, then unfreeze the reused layers, lower the learning rate, and keep training to fine-tune them. The more similar the tasks, the more layers you can reuse; lower layers transfer best.

## 8. Write the momentum optimization update, explain why it speeds up training, and say what Nesterov accelerated gradient changes.

Plain gradient descent updates θ ← θ − η∇J(θ), ignoring earlier gradients. Momentum uses the gradient as an acceleration rather than a speed:

1. m ← βm − η∇J(θ)
2. θ ← θ + m

The momentum coefficient β (typically 0.9) acts as friction, from 0 (high friction) to 1 (none). With a constant gradient, the steps grow to a terminal size of η‖∇J(θ)‖ / (1 − β), so with β = 0.9 it goes 10 times faster than plain gradient descent: it escapes plateaus and rolls down long, narrow valleys much faster.

Nesterov accelerated gradient (NAG) measures the gradient slightly ahead, at θ + βm, instead of at θ. The momentum generally points toward the optimum, so that gradient is a bit more accurate, and it damps oscillations: NAG is almost always faster than plain momentum. In PyTorch: torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True).

## 9. Walk through the Adam update rule, what its bias-correction steps are for, and its usual hyperparameter values.

Adam (adaptive moment estimation) tracks decaying averages of past gradients (the first moment, as in momentum) and of past squared gradients (the uncentered second moment, as in RMSProp). With g = ∇J(θ) and iteration t = 1, 2, …:

1. m ← β₁m − (1 − β₁)g
2. s ← β₂s + (1 − β₂)g ⊗ g
3. m̂ ← m / (1 − β₁ᵗ)
4. ŝ ← s / (1 − β₂ᵗ)
5. θ ← θ + η m̂ ⊘ (√ŝ + ε)

Dividing by √ŝ gives each parameter its own effective learning rate, smaller where gradients are large. Since m and s start at 0, they're biased toward 0 early on; steps 3 and 4 scale them up to compensate, an effect that fades as t grows. torch.optim.Adam's defaults are the usual values: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, η = 0.001, and the learning rate needs less tuning than with plain SGD.

## 10. Why use AdamW rather than Adam with ℓ2 regularization, and how do you keep weight decay off bias and normalization parameters?

Weight decay shrinks the weights slightly at each step (e.g., multiplying them by 0.99). With SGD it's equivalent to ℓ2 regularization, so torch.optim.SGD(..., weight_decay=1e-4) is the easy way to get ℓ2. With Adam it isn't: the ℓ2 gradient gets rescaled by Adam's per-parameter scaling like any other gradient, and Adam with ℓ2 tends to generalize worse. AdamW decouples the weight decay from the gradient update and applies it directly to the weights: use torch.optim.AdamW and tune weight_decay.

The weight_decay argument applies to every parameter, including biases and batch-norm or layer-norm parameters, where it adds little regularization and can hurt training. Parameter groups let you set hyperparameters per group of parameters:

```python
no_decay = [p for n, p in model.named_parameters() if "bias" in n or "bn" in n]
decay = [p for n, p in model.named_parameters() if "bias" not in n and "bn" not in n]
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-4},
    {"params": no_decay, "weight_decay": 0.0},  # AdamW's default is 0.01
], lr=1e-3)
```

The name test assumes your batch-norm modules have "bn" in their names.

## 11. How do you wire a learning-rate scheduler into a PyTorch training loop, and what's different about ReduceLROnPlateau?

Wrap the optimizer in a scheduler from torch.optim.lr_scheduler, then call scheduler.step() once per epoch, after that epoch's optimizer steps:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        ...  # forward pass, loss.backward(), optimizer.step(), optimizer.zero_grad()
    scheduler.step()
```

ExponentialLR multiplies the learning rate by gamma at each step: with gamma=0.9, it's down to about 35% of its initial value after 10 epochs.

ReduceLROnPlateau (performance scheduling) reacts to a metric rather than to the epoch count. To track validation accuracy, create it with mode="max" (the default, "min", suits a loss), plus e.g. patience=2 and factor=0.1; evaluate on the validation set at the end of each epoch and call scheduler.step(val_metric). It multiplies the learning rate by factor once the metric has failed to improve for more than patience epochs in a row.

## 12. Compare cosine annealing, cosine annealing with warm restarts, learning-rate warmup, and 1cycle scheduling: what does each do, and when is it useful?

- Cosine annealing (CosineAnnealingLR): η_t = η_min + ½(η_max − η_min)(1 + cos(πt / T_max)) falls from η_max at epoch 0 to η_min at epoch T_max while staying fairly high for most of training. It generally beats exponential decay, but you must pick T_max and η_min up front.
- Warm restarts (CosineAnnealingWarmRestarts): repeats the cosine cycle, often doubling its length each round (T_mult=2). Each jump back up helps escape plateaus and local optima.
- Warmup (e.g., LinearLR with start_factor=0.1, end_factor=1.0, total_iters=3): ramps the rate up over the first few epochs, taming the chaotic start of sensitive models such as RNNs, or of very large batches.
- 1cycle (OneCycleLR, stepped after every batch): the rate climbs to a peak, comes back down, then drops far lower for the final epochs, while momentum moves the opposite way (e.g., 0.95 → 0.85 → 0.95). It can speed up training considerably.

## 13. How does dropout work during training versus inference, and why does it reduce overfitting?

At every training step, each neuron except the output neurons is dropped (outputs 0) with probability p, the dropout rate, typically 10% to 50%. Neurons can't co-adapt with their neighbors or lean on a few inputs, so each must be useful on its own, making the network more robust. Equivalently, each step trains one of 2ᴺ possible weight-sharing subnetworks (for N droppable neurons), and the final network acts like an averaging ensemble of them.

Nothing is dropped after training, so each neuron would suddenly get about 1/(1 − p) times more input than during training. To compensate, the kept inputs are divided by the keep probability 1 − p during training. nn.Dropout(p=0.2) does exactly this in training mode and passes inputs through unchanged in evaluation mode, so model.train() and model.eval() matter. Since dropout is off at evaluation, compare the validation loss with a training loss measured without dropout.
