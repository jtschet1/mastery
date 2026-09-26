# Chapter 13 — Processing Sequences Using RNNs and CNNs

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Write the output equation of a layer of simple recurrent neurons for a whole mini-batch, and explain each term and its shape.

At time step t, the layer combines the current inputs with its own previous outputs:

Ŷ₍ₜ₎ = φ(X₍ₜ₎Wₓ + Ŷ₍ₜ₋₁₎Wŷ + b)

- X₍ₜ₎ (m × n_inputs): the inputs at step t for the m instances in the batch.
- Wₓ (n_inputs × n_neurons): the weights for the current inputs.
- Ŷ₍ₜ₋₁₎ (m × n_neurons): the outputs at the previous step, all zeros at t = 0.
- Wŷ (n_neurons × n_neurons): the recurrent weights.
- b (size n_neurons): the bias vector; φ is the activation function (tanh by default in nn.RNN).

Stacking Wₓ on top of Wŷ gives one W of shape (n_inputs + n_neurons) × n_neurons, so Ŷ₍ₜ₎ = φ([X₍ₜ₎ Ŷ₍ₜ₋₁₎]W + b), with the two matrices concatenated horizontally. The same W and b are used at every step, and because Ŷ₍ₜ₋₁₎ depends on earlier steps, Ŷ₍ₜ₎ depends on all inputs since t = 0.

## 2. How does backpropagation through time (BPTT) train a recurrent network?

It unrolls the RNN through time, making it a feedforward network with one copy of the cell per time step, then runs ordinary backpropagation:

1. Forward pass through the unrolled network, producing an output at each step.
2. Compute the loss on the output sequence. It may ignore some outputs; a sequence-to-vector model only uses the last one.
3. Backpropagate. Gradients flow only through the outputs the loss uses, then back through earlier steps via the hidden states.
4. W and b are shared by every step, so each parameter's gradient sums the contributions from all the steps. One gradient descent step then updates them.

PyTorch's autograd does all this, whether you loop over the time steps yourself or call nn.RNN. The catch: the unrolled network is as deep as the sequence is long, so long sequences train slowly and are prone to unstable gradients.

## 3. What is naive forecasting, and which metrics does the chapter use to compare forecasts against it?

Naive forecasting copies a past value as the forecast, usually the latest one (tomorrow = today). For the Chicago ridership data, copying the value from 7 days earlier works better because the series has a strong weekly seasonality: it's highly correlated with a one-week-lagged copy of itself (autocorrelation). It needs no training and is often surprisingly hard to beat, so it sets the bar every model must clear.

- MAE (mean absolute error): the average miss, in the series' own units (riders).
- MAPE (mean absolute percentage error): each error divided by the target value, so it's comparable across series of different scales. With naive forecasts, rail had the lower MAE but the higher MAPE, simply because bus ridership is larger.
- MSE (mean squared error): penalizes large errors quadratically, so prefer it when big misses hurt disproportionately.

## 4. Explain the ARMA forecasting equation, and what ARIMA and SARIMA add to it.

ARMA forecasts the next value as a weighted sum of recent values plus a weighted sum of recent forecast errors:

ŷ₍ₜ₎ = ∑ (i = 1 to p) αᵢ·y₍ₜ₋ᵢ₎ + ∑ (i = 1 to q) θᵢ·ε₍ₜ₋ᵢ₎, where ε₍ₜ₎ = y₍ₜ₎ − ŷ₍ₜ₎

The first sum is the autoregressive part (the last p values, learned weights αᵢ); the second is the moving-average part (the last q forecast errors, learned weights θᵢ). It assumes the series is stationary.

- ARIMA first differences the series d times (the order of integration): one round turns a linear trend into a constant, and d rounds remove polynomial trends up to degree d. It applies ARMA to the result, then adds back what differencing subtracted.
- SARIMA also models a seasonal pattern of period s: P, D and Q play the roles of p, d and q at lags s, 2s, 3s, and so on (seven hyperparameters in total).

## 5. How do you fit a SARIMA model with statsmodels and forecast the next day, and how does the chapter evaluate and tune it?

The ARIMA class covers the whole ARMA family. Unlike Scikit-Learn, you pass the data to the constructor, not to fit():

```python
from statsmodels.tsa.arima.model import ARIMA

rail_series = df.loc["2019-01-01":"2019-05-31"]["rail"].asfreq("D")
model = ARIMA(rail_series,
              order=(1, 0, 0),              # p, d, q
              seasonal_order=(0, 1, 1, 7))  # P, D, Q, s
model = model.fit()
y_pred = model.forecast()  # forecast for the next day
```

asfreq("D") sets the daily frequency explicitly; otherwise statsmodels has to guess it and warns. One day's forecast proves little, so the chapter loops over every day from March to May 2019, refitting on the data up to that day and forecasting the next one, then computes the MAE over the period (it clearly beats naive forecasting). To pick hyperparameters, grid search small values (p, q, P, Q usually 0 to 2; d and D usually 0 or 1; s is the main seasonal period, 7 here) and keep the lowest MAE. ACF/PACF analysis or the AIC/BIC are more principled alternatives.

## 6. Sketch the chapter's TimeSeriesDataset, and explain how it turns one long series into training batches for an RNN.

Every run of window_length consecutive values becomes an input, and the value right after it becomes the target:

```python
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, series, window_length):
        self.series = series  # shape [length, n_features]
        self.window_length = window_length

    def __len__(self):
        return len(self.series) - self.window_length

    def __getitem__(self, idx):
        end = idx + self.window_length  # first index after the window
        return self.series[idx:end], self.series[end]
```

Keep the series 2D even when it's univariate (the chapter uses df[["rail"]] rather than df["rail"]), so a DataLoader yields batches of shape [batch, window_length, n_features], which is what a recurrent layer with batch_first=True expects. The training loader uses shuffle=True, which shuffles whole windows (so batches are closer to IID), not the values inside them. Split the training, validation and test sets across time, and scale the values to roughly the 0–1 range (the chapter divides ridership by one million).

## 7. Sketch the chapter's sequence-to-vector forecaster built on nn.RNN, and explain what nn.RNN returns.

An nn.RNN layer, then a linear head applied to the last time step's output:

```python
class SimpleRnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):  # X: [batch, time, input_size]
        outputs, last_state = self.rnn(X)
        return self.output(outputs[:, -1])
```

nn.RNN returns two tensors:
- outputs: the top layer's hidden state at every time step, [batch, time, hidden_size].
- last_state: each layer's hidden state after the final step, [num_layers, batch, hidden_size]. batch_first doesn't apply to it.

The head is needed for two reasons: the hidden state has hidden_size values while the target has output_size, and tanh (nn.RNN's default activation) only outputs values between −1 and 1, while the targets can exceed 1. nn.RNN starts from a zero hidden state and uses cuDNN's optimized kernels on Nvidia GPUs, so it's much faster than a hand-written loop over time steps.

## 8. How can a model trained to forecast one step ahead produce a 14-day forecast, and what's the catch?

Use it autoregressively: forecast the next value, append the forecast to the inputs as if it had actually happened, and repeat 14 times:

```python
model.eval()
with torch.no_grad():
    X = rail_valid[:56].unsqueeze(dim=0)  # [1, 56, 1]: one 56-day window
    for step_ahead in range(14):
        y_pred_one = model(X)             # [1, 1]
        X = torch.cat([X, y_pred_one.unsqueeze(dim=1)], dim=1)  # add a time step
    Y_pred = X[0, -14:, 0]                # the 14 forecasts
```

The catch is that errors accumulate: each mistake becomes an input to every later forecast, so this only works well for a few steps. The alternative is to train the model to output all 14 values in one shot (each target becomes the vector of the next 14 values), which avoids compounding errors. You can also combine the two: forecast 14 days at once, append them to the inputs, and run the model again for the following 14.

## 9. Why train a sequence-to-sequence forecaster when you only use its last time step's forecast, and how does the chapter build its targets?

At every time step the model forecasts the next 14 values, so the loss gets a term from every step instead of only the last. That means many more error gradients, and they don't have to flow back through as many time steps, which stabilizes and speeds up training. Since it forecasts from inputs of every length, it may also overfit less to the training window length. It isn't cheating, even though the targets overlap the inputs: an RNN is causal, so each output only depends on past inputs. After training, keep only the last step's forecasts, Y_preds[:, -1].

Tensor.unfold() builds the per-step targets as sliding windows:

```python
target_period = self.series[idx + 1 : end + 14, 0]  # rail column only
target = target_period.unfold(dimension=0, size=14, step=1)  # [window_length, 14]
```

The nn.Linear head is applied to the full [batch, time, hidden] output: it acts on the last dimension, so it runs at every time step and returns [batch, time, 14].

## 10. Write the GRU cell's equations, and explain how the GRU simplifies the LSTM cell.

For one instance at time step t:

- z₍ₜ₎ = σ(W_xzᵀ x₍ₜ₎ + W_hzᵀ h₍ₜ₋₁₎ + b_z)
- r₍ₜ₎ = σ(W_xrᵀ x₍ₜ₎ + W_hrᵀ h₍ₜ₋₁₎ + b_r)
- g₍ₜ₎ = tanh(W_xgᵀ x₍ₜ₎ + W_hgᵀ (r₍ₜ₎ ⊗ h₍ₜ₋₁₎) + b_g)
- h₍ₜ₎ = z₍ₜ₎ ⊗ h₍ₜ₋₁₎ + (1 − z₍ₜ₎) ⊗ g₍ₜ₎

Compared with the LSTM:
- One state vector h₍ₜ₎ replaces the short-term and long-term states.
- One gate controller, z₍ₜ₎, is both the forget and the input gate: near 1 it keeps the old state and ignores the candidate g₍ₜ₎; near 0 it erases the old content and writes g₍ₜ₎. Memory is erased exactly where new memory is stored.
- No output gate: the full state is output at every step. Instead, r₍ₜ₎ controls how much of the previous state the main layer g₍ₜ₎ sees.

It often performs as well as an LSTM, with fewer parameters. PyTorch provides nn.GRU and, for hand-written loops, nn.GRUCell.

## 11. What input shape does nn.Conv1d expect, and what must you adjust when a Conv1d layer feeds a GRU, as in the chapter's DownsamplingModel?

nn.Conv1d expects [batch, channels, length], with the features as channels, while a recurrent layer with batch_first=True uses [batch, length, features]. So you permute before and after the convolution:

```python
def forward(self, X):
    Z = X.permute(0, 2, 1)  # [batch, features, time]
    Z = self.conv(Z)        # nn.Conv1d(input_size, 32, kernel_size=4, stride=2)
    Z = Z.permute(0, 2, 1)  # back to [batch, time, 32]
    Z = torch.relu(Z)
    Z, _states = self.gru(Z)
    return self.linear(Z)   # 14 forecasts at each remaining time step
```

The convolution also changes the sequence length, and the targets must line up with the outputs. With stride 1 and "same" padding the length is unchanged, but this layer uses no padding ("valid", the default) and a stride of 2, which roughly halves it. Its first output sees input steps 0 to 3, so it must forecast steps 4 to 17: the chapter drops the first 3 per-step targets and keeps every second one after that (target[3::2]). The shorter sequence helps the GRU capture longer patterns, so the window was doubled to 112 days.

## 12. How does a WaveNet-style stack of dilated causal convolutions work, and how does the chapter implement the causal part?

WaveNet stacks 1D convolutional layers (kernel size 2) whose dilation rate, the spacing between each neuron's inputs, doubles at every layer: 1, 2, 4, 8, and so on. The receptive field doubles too: 2 time steps after the first layer, 4 after the second, 1,024 after ten layers (dilations 1 to 512). Lower layers learn short-term patterns and higher layers long-term ones, with few parameters and no recurrence, so it can handle audio with tens of thousands of steps per second. The paper repeated that 10-layer block three times; the chapter uses dilations (1, 2, 4, 8) twice, each conv followed by ReLU, then an nn.Linear head.

To stay causal (never peeking at future steps) while keeping the sequence length, each layer pads only on the left, by (kernel_size − 1) × dilation:

```python
class CausalConv1d(nn.Conv1d):
    def forward(self, X):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        X = F.pad(X, (padding, 0))  # (left, right) padding of the time axis
        return super().forward(X)
```
