# Chapter 10 — Building Neural Networks with PyTorch

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. When moving data between NumPy and PyTorch, what dtype and memory-sharing gotchas should you watch for?

- NumPy floats default to 64 bits, PyTorch's to 32 bits, which is plenty for neural nets and uses half the RAM. torch.tensor(array) keeps a float64 array's dtype, which wastes memory and time and clashes with float32 model weights (you get a dtype mismatch error). Pass dtype=torch.float32, or use torch.FloatTensor(array), which converts to 32 bits.
- torch.tensor() and torch.FloatTensor() copy the data. torch.from_numpy(array) creates a CPU tensor that shares the array's memory: no copy, but modifying one modifies the other.
- Going the other way, tensor.numpy() only works on a CPU tensor that doesn't require gradients, so the general form is tensor.detach().cpu().numpy().

## 2. Sketch linear regression trained with raw tensors and autograd (no nn.Module, no optimizer), and explain what each autograd step does.

This runs batch gradient descent on the whole training set:

```python
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)
for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

- requires_grad=True makes w and b leaf tensors that autograd tracks. Each forward pass builds a fresh computation graph on the fly: every result computed from them carries a grad_fn recording the operation that produced it.
- loss.backward() backpropagates from the loss to the leaves and adds each gradient to the leaf's grad attribute.
- The update runs inside torch.no_grad() so it isn't recorded in the graph; an in-place update of a leaf that requires grad would raise an error anyway.
- backward() accumulates gradients rather than overwriting them, so zero them every iteration, or the updates will be silently wrong.
- y_train must be a column vector of shape [m, 1], like y_pred. With shape [m], the subtraction would broadcast to [m, m] and quietly compute the wrong loss.

## 3. How would you write an evaluation function that computes the RMSE correctly over mini-batches, and how do TorchMetrics streaming metrics help?

Averaging a per-batch metric can be wrong: the mean of the batch RMSEs isn't the RMSE over the whole dataset, because the mean of square roots isn't the square root of the mean. You could average an additive metric, like the MSE, and take the square root at the end, or use a TorchMetrics streaming metric, which accumulates what it needs across batches: reset() at the start, update(predictions, targets) for each batch, and compute() at the end. An evaluation function built on one:

```python
def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            metric.update(model(X_batch), y_batch)
    return metric.compute()

rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
valid_rmse = evaluate_tm(model, valid_loader, rmse)
```

Note the model.eval() call and the torch.no_grad() context, and that the metric is moved to the same device as the model.

## 4. Sketch a custom nn.Module for a nonsequential Wide & Deep network, and explain how PyTorch finds its parameters.

Create the layers in the constructor after calling super().__init__(), and wire them together in forward(), which can use any Python logic:

```python
class WideAndDeep(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU())
        self.output_layer = nn.Linear(n_features + 40, 1)

    def forward(self, X):
        deep_output = self.deep_stack(X)
        wide_and_deep = torch.concat([X, deep_output], dim=1)
        return self.output_layer(wide_and_deep)
```

Assigning a module or an nn.Parameter to an attribute registers it, so model.parameters(), which you pass to the optimizer, finds every parameter recursively, including those inside deep_stack. Plain tensors aren't included, even with requires_grad=True. Modules or parameters kept in a regular Python list or dict aren't registered either: use nn.ModuleList or nn.ModuleDict (nn.ParameterList or nn.ParameterDict for parameters). Call model(X) rather than model.forward(X), so that hooks run.

## 5. What does a DataLoader need from a dataset, and how would you write a custom Dataset that feeds named inputs to a multi-input model?

A DataLoader only needs an object with __len__() (the number of samples) and __getitem__(idx) (one sample with its target). TensorDataset provides this for tensors sharing the same first dimension: for example, TensorDataset(X_wide, X_deep, y) yields three tensors per batch. For anything else, subclass torch.utils.data.Dataset. With several inputs, returning a dictionary of named inputs avoids mixing up their order. Here's such a dataset, then the training loop's use of it:

```python
class WideAndDeepDataset(torch.utils.data.Dataset):
    def __init__(self, X_wide, X_deep, y):
        self.X_wide, self.X_deep, self.y = X_wide, X_deep, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        inputs = {"X_wide": self.X_wide[idx], "X_deep": self.X_deep[idx]}
        return inputs, self.y[idx]

for inputs, y_batch in train_loader:
    inputs = {name: X.to(device) for name, X in inputs.items()}
    y_pred = model(**inputs)  # keys match forward()'s argument names
```

The DataLoader's default collation batches each dictionary entry separately, so each batch arrives as a dictionary of batched tensors.

## 6. Sketch loading Fashion MNIST with TorchVision as float tensors, with a validation split. What does each piece do?

This loads the 60,000 training images and holds out 5,000 of them for validation:

```python
import torchvision
import torchvision.transforms.v2 as T

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])
```

- The transform runs on the fly each time a sample is accessed. Without it you'd get PIL images with integer pixels from 0 to 255. ToImage converts them to TorchVision's Image type (a Tensor subclass) with the channel dimension first, and ToDtype(torch.float32, scale=True) converts to floats scaled to 0–1.
- root is where the data is stored, download=True fetches it if it's missing, and train=False would load the test set instead.
- random_split() carves out a validation set; call torch.manual_seed() first for a reproducible split.

Each image is then a [1, 28, 28] tensor (channels, height, width): PyTorch puts channels first, unlike Matplotlib or PIL. The class names are in the original dataset's classes attribute; the subsets returned by random_split() don't have it.

## 7. Sketch a hyperparameter search with Optuna. What do the key calls do, and how can you stop hopeless trials early?

This tunes an MLP classifier's learning rate and hidden layer size:

```python
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(n_inputs=28 * 28, n_hidden1=n_hidden,
                            n_hidden2=n_hidden, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ...  # train the model, then measure its validation accuracy
    return validation_accuracy

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)
```

- The objective asks the Trial for values with suggest_float(), suggest_int(), or suggest_categorical(); log=True samples on a log scale, so small values get explored too. It returns the validation score.
- Optuna minimizes by default, hence direction="maximize". The default TPE sampler learns from past trials to focus on promising regions, which usually beats random search. Results end up in study.best_params and study.best_value.
- To pass data loaders without globals, wrap the objective with functools.partial() or a lambda.
- To prune bad trials, pass a pruner such as MedianPruner to create_study(), call trial.report(score, epoch) after each epoch, and raise optuna.TrialPruned() when trial.should_prune() returns True.

## 8. What's the recommended way to save and load a PyTorch model, and why is it better than torch.save(model)?

torch.save(model, path) pickles the whole model object. That's convenient, but loading a pickle can run arbitrary code, so only load trusted files (torch.load() needs weights_only=False for this). Pickles are also brittle across Python versions and folder layouts, and your custom classes must be importable at load time.

Saving model.state_dict() is safer: it's an OrderedDict of the parameters (as named by named_parameters()) plus any buffers registered with register_buffer(), and loading it with weights_only=True only accepts data. You must rebuild the exact same architecture first, so save its hyperparameters alongside:

```python
torch.save({"model_state_dict": model.state_dict(),
            "model_hyperparameters": {"n_inputs": 28 * 28, "n_hidden1": 300,
                                      "n_hidden2": 100, "n_classes": 10}},
           "my_model.pt")

loaded = torch.load("my_model.pt", weights_only=True)
new_model = ImageClassifier(**loaded["model_hyperparameters"])
new_model.load_state_dict(loaded["model_state_dict"])
new_model.eval()
```

To resume training later, also save the optimizer's state_dict() and information like the current epoch.
