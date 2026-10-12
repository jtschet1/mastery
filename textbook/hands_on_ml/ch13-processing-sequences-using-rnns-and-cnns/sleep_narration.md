# Chapter 13 — Processing Sequences Using RNNs and CNNs, to fall asleep to

Chapter thirteen. Processing sequences using recurrent neural networks and convolutional neural networks.

---

Recurrent neural networks, or RNNs, are a class of networks that can predict the future, up to a point. They can analyze time series, such as the daily active users of a website, the hourly temperature in a city, a home's daily power consumption, or the trajectories of nearby cars. Once an RNN has learned past patterns in the data, it can forecast the future, assuming those patterns still hold.

More generally, RNNs work on sequences of arbitrary length rather than on fixed-size inputs. They can take sentences, documents, or audio samples as input, which makes them useful for natural language processing tasks such as automatic translation and speech-to-text.

RNNs face two main difficulties. The first is unstable gradients, which can be alleviated with techniques including recurrent dropout and recurrent layer normalization. The second is a very limited short-term memory, which can be extended with long short-term memory cells and gated recurrent unit cells. RNNs are not the only networks for sequences: for small sequences a regular dense network can do the job, and for very long sequences, such as audio or text, convolutional neural networks work well too.

---

Recurrent neurons and layers. In a feedforward network, activations flow in one direction only, from the input layer to the output layer. A recurrent neural network looks much the same, except that it also has connections pointing backward.

The simplest RNN is a single neuron that receives inputs, produces an output, and sends that output back to itself. At each time step, also called a frame, the recurrent neuron receives the current inputs plus its own output from the previous time step. At the first time step there is no previous output, so it is generally set to zero. Drawing the same neuron once per time step, along a time axis, is called unrolling the network through time.

In a layer of recurrent neurons, every neuron receives at each time step both the input vector and the layer's output vector from the previous time step. Each recurrent neuron has two sets of weights: one for the inputs and one for the previous outputs. For the whole layer, these form two weight matrices.

The layer's output at a time step is the activation function applied to the sum of three terms: the current inputs times the input weight matrix, the previous outputs times the recurrent weight matrix, and the bias vector. For a mini-batch, the inputs at one time step form a matrix with one row per instance, and so do the outputs. The input weight matrix has one row per input feature and one column per neuron, and the recurrent weight matrix is square, with one row and one column per neuron. The two are often stacked into a single weight matrix, applied to the inputs and previous outputs placed side by side.

The output at a time step depends on the outputs of the previous step, which depend on the step before, and so on, so it is a function of all the inputs since the first time step.

The idea of backward connections dates back to the origins of artificial neural networks, but the first modern RNN architecture was proposed by Michael I. Jordan in 1986. It fed its own outputs from the previous time step back in, which is called output feedback.

---

Memory cells. Since a recurrent neuron's output depends on all the previous inputs, it has a form of memory. A part of a neural network that preserves some state across time steps is called a memory cell, or simply a cell. A single recurrent neuron, or a layer of them, is a very basic cell that can learn only short patterns, typically about ten steps long, depending on the task. More complex cells can learn patterns roughly ten times longer.

A cell's state at a time step, called its hidden state, is a function of the current inputs and of its state at the previous time step. Its output is also a function of the previous state and the current inputs, and typically it is just a function of the current state. In basic cells the output equals the state, but in more complex cells this is not always the case. Feeding back the hidden state rather than the outputs was proposed by Jeffrey Elman in 1990. It is called state feedback, and it is the most common approach today.

---

Input and output sequences. A sequence-to-sequence network takes a sequence of inputs and produces a sequence of outputs. It is useful for forecasting time series, such as a home's daily power consumption: it is fed the data for the last N days and trained to output the power consumption shifted one day into the future, from N minus one days ago to tomorrow.

A sequence-to-vector network takes a sequence of inputs and ignores all outputs except the last one. For example, it can read the words of a movie review and output a sentiment score, from zero for hate to one for love.

A vector-to-sequence network is fed the same input vector at every time step and outputs a sequence. For example, the input could be an image, or the output of a CNN, and the output a caption for that image.

An encoder-decoder is a sequence-to-vector network, called the encoder, followed by a vector-to-sequence network, called the decoder. For translation, the encoder converts a sentence into a single vector representation, and the decoder turns that vector into a sentence in another language. This works much better than translating on the fly with a single sequence-to-sequence RNN, because the last words of a sentence can affect the first words of its translation, so the whole sentence must be seen before translating it.

---

Training RNNs. An RNN is trained by unrolling it through time and then using regular backpropagation. This is called backpropagation through time, or BPTT.

A forward pass runs through the unrolled network, and the output sequence is evaluated with a loss function of all the targets and all the predictions. The loss may ignore some outputs: a sequence-to-vector RNN ignores all but the last one. The gradients of the loss then flow backward through the unrolled network, but only through the outputs that the loss uses. Since the same weights and biases are used at every time step, their gradients are tweaked several times during backpropagation. Once all the gradients are computed, a regular gradient descent step updates the parameters. PyTorch handles all of this.

---

Forecasting a time series. The running example is forecasting how many passengers will ride Chicago Transit Authority buses and trains the next day, using daily ridership data since 2001. The data is loaded with Pandas, sorted by date, stripped of the redundant total column, which is just bus plus rail, and of duplicate rows. A day type column holds W for weekdays, A for Saturdays, and U for Sundays and holidays.

A time series is data with values at different time steps, usually at regular intervals. With several values per time step, such as bus and rail, it is a multivariate time series. With a single value per time step, such as bus alone, it is a univariate time series. Forecasting future values is the most typical task. Others include imputation, which fills in missing past values, classification, and anomaly detection.

Plotting March to May 2019 shows a similar pattern repeated every week. This is called a weekly seasonality, and here it is so strong that forecasting tomorrow's ridership by copying the value from a week earlier gives reasonably good results. Copying a past value to make a forecast is called naive forecasting. It usually means copying the latest known value, but with a strong weekly seasonality, the value from one week earlier works better. Naive forecasting is often a great baseline, and it can be tricky to beat.

Overlaying each series with a copy lagged by one week shows how closely the two track each other. A time series that is correlated with a lagged version of itself is said to be autocorrelated. Subtracting, at each time step, the value from seven days earlier is called differencing. The differences are mostly small, except at the end of May, which turns out to be the Memorial Day long weekend; the day type column could help forecast such holidays.

Over March to May 2019, the naive forecasts have a mean absolute error, or MAE, of about forty-four thousand bus riders and forty-two thousand rail riders. Dividing each error by the target value gives the mean absolute percentage error, or MAPE: about eight point three percent for bus and nine percent for rail. The rail forecasts have the lower MAE, but the bus forecasts have the lower MAPE. Bus ridership is larger, so its absolute errors are naturally larger, but relative to the targets, the bus forecasts are slightly better.

The MAE, the MAPE, and the mean squared error, or MSE, are among the most common metrics for evaluating forecasts. The right choice depends on the task. If large errors are disproportionately costly, the MSE may be preferable, since it strongly penalizes them.

---

Seasonality, trends, and differencing. Monthly averages from 2001 to 2019, leaving out more recent data to reduce the risk of data snooping, plus a twelve-month rolling average to show long-term trends, reveal a yearly seasonality as well. It is noisier than the weekly seasonality and more visible for rail than for bus, with peaks and troughs at about the same dates each year.

Differencing over twelve months removes not only the yearly seasonality but also the long-term trends. For example, the downward linear trend from 2016 to 2019 becomes a roughly constant negative value. Differencing is a common way to remove trend and seasonality, because a stationary time series, one whose statistical properties stay the same over time, is easier to study. Forecasts made on the differenced series are turned back into forecasts for the original series by adding back the past values that were subtracted.

Long-term patterns matter less than short-term ones for next-day forecasts, but accounting for them can still improve accuracy slightly. For example, daily bus ridership dropped by about two thousand five hundred in October 2017, which represents about five hundred seventy fewer passengers each week. At the end of October 2017, it would make sense to forecast by copying last week's value and subtracting five hundred seventy.

---

The ARMA model family. The autoregressive moving average model, or ARMA, was developed by Herman Wold in the 1930s. Its forecast is a weighted sum of the series' past p values, plus a weighted sum of the past q forecast errors, where each forecast error is the actual value minus the forecast. The first sum is the autoregressive component: it performs regression on past values. The second sum is the moving average component. The weights in both sums are learned. The numbers p and q are hyperparameters, and p sets how far into the past the model looks.

ARMA assumes the time series is stationary. If it is not, differencing can help. Differencing over a single time step approximates the derivative of the series, giving its slope at each time step, so it eliminates any linear trend. The series three, five, seven, nine, eleven becomes two, two, two, two. A quadratic trend needs two rounds: the series one, four, nine, sixteen, twenty-five, thirty-six becomes three, five, seven, nine, eleven after one round, and two, two, two, two after the second. In general, d rounds of differencing approximate the derivative of order d, and eliminate polynomial trends up to degree d. The hyperparameter d is called the order of integration.

Differencing is the central contribution of the autoregressive integrated moving average model, or ARIMA, introduced in 1970 by George Box and Gwilym Jenkins. It runs d rounds of differencing to make the series more stationary, then applies a regular ARMA model. When forecasting, it adds back the terms that differencing subtracted.

The seasonal ARIMA model, or SARIMA, models the series the same way as ARIMA, but it also models a seasonal component with a given period, such as weekly, using the same approach. It has seven hyperparameters: p, d, and q, as in ARIMA; capital P, D, and Q, which work like their lowercase counterparts but on the values one period back, two periods back, and so on; and the period of the seasonal pattern, called s.

---

Fitting a SARIMA model. The statsmodels library implements the ARMA family in its ARIMA class. The book fits it to the rail ridership from the start of 2019 up to the last day of May 2019, treated as today, with the series' frequency set to daily. The hyperparameters are p equal to one, d and q equal to zero, capital P equal to zero, capital D and Q equal to one, and a period s of seven. Unlike Scikit-Learn, statsmodels receives the data when the model is created, rather than when it is fit.

The forecast for June first, 2019, is about four hundred twenty-eight thousand passengers, but the actual figure was about three hundred seventy-nine thousand: an error of twelve point nine percent, slightly worse than the naive forecast's twelve point six percent. To check whether that day was just unlucky, the same process is repeated for every day from March to May, retraining the model each day on all the data up to that day. The resulting MAE is about thirty-two thousand, much better than the naive forecast's forty-two thousand.

The simplest way to choose SARIMA hyperparameters is a grid search: evaluate each combination the same way, and keep the one with the lowest MAE, or with another metric that better matches the business objective. Good values of p, q, capital P, and capital Q are usually small, from zero to two, sometimes up to five or six. The orders d and capital D are usually zero or one, sometimes two. The period s is that of the main seasonal pattern, here seven. More principled methods analyze the autocorrelation function and the partial autocorrelation function, or minimize the AIC or BIC metrics, which penalize models with too many parameters and reduce the risk of overfitting.

---

Preparing the data for machine learning models. With two baselines in place, naive forecasting and SARIMA, the goal is to forecast tomorrow's ridership from the past eight weeks, which is fifty-six days. Each input is a sequence of fifty-six values, from time step t minus fifty-five to time step t, and the output is a single value: the forecast for time step t plus one. The training data is every fifty-six-day window from the past, with the value immediately after each window as its target.

A custom PyTorch dataset chops a time series into all the windows of a given length, each paired with its target. For the series zero through five and a window length of three, the windows are zero, one, two, with target three; one, two, three, with target four; and two, three, four, with target five. PyTorch's recurrent modules expect each sequence to be two-dimensional, sequence length by dimensionality, so a univariate series has a dimensionality of one.

The rail data is split into a training period from 2016 to 2018, a validation period from January to May 2019, and a test period from June 2019 on. The values are divided by one million, so they end up near the range zero to one, which suits the default weight initialization and learning rate.

Time series are generally split across time. Splitting along another dimension, such as by company in financial data on ten thousand companies, gives a longer training period, but correlated companies in both the training and test sets make the test estimate optimistic.

Gradient descent expects independent and identically distributed training instances, so the training data loader shuffles the windows, though not their contents, in batches of thirty-two.

---

Forecasting using a linear model. A basic linear model flattens each window and passes it to a single linear layer with fifty-six inputs and one output. The flatten layer is needed because the inputs are shaped batch size by window length by dimensionality, while the linear layer expects batch size by features. The model is trained with the Huber loss, which usually works better than minimizing the MAE directly, using stochastic gradient descent with momentum. It reaches a validation MAE of about thirty-seven thousand seven hundred: better than naive forecasting, but worse than SARIMA.

---

Forecasting using a simple RNN. The book first builds a simple RNN by hand. Its memory cell is a linear layer followed by the hyperbolic tangent activation, tanh, applied at every time step to the current inputs concatenated with the hidden state. A final linear layer turns the last hidden state into the forecast. The input size is one, for a univariate series. The hidden size, which is the number of recurrent neurons, is thirty-two, a hyperparameter that can be tuned. The output size is one.

Tanh is common in RNNs because it tends to be more stable there than other activation functions. The output layer is needed for two reasons: the hidden state has thirty-two dimensions while the target has one, and tanh only outputs values between minus one and one, while the values to forecast occasionally exceed one.

The forward pass initializes the hidden state to zeros, one zero per recurrent neuron for each window in the batch. It swaps the batch and time dimensions, so it can loop over the time steps. At each step, it concatenates that step's inputs with the hidden state and passes the result through the memory cell, which gives the new hidden state. After the last step, the final hidden state goes through the output layer. All the sequences in the batch are processed simultaneously. This is a sequence-to-vector model.

It reaches a validation MAE of about thirty thousand seven hundred, the best so far, beating SARIMA. The series was only scaled, without removing trend or seasonality, which makes it quick to search for promising models without much preprocessing. For the best performance, making the series more stationary, for example by differencing, may help.

---

PyTorch's RNN module. PyTorch's RNN module does the same job in much less code. It is created with the input size and the hidden size, plus a batch-first flag, because by default it expects the time dimension to come first. Internally, it initializes the hidden state to zeros and processes each time step with a simple memory cell, a linear layer and an activation function, tanh by default.

The module returns two things. The first is the outputs of the top recurrent layer at every time step, shaped batch size by window length by hidden size; for a simple RNN, these are just the hidden states of the top layer. The second is the hidden state of each recurrent layer after the last time step, shaped number of layers by batch size by hidden size. The model takes the last output, which is also the last state of the top layer, and passes it through a linear output layer.

The results are similar to the hand-built version, but training is generally much faster, because the module is well optimized, and on Nvidia GPUs it uses the highly optimized cuDNN library. The module has two bias vectors, one for the inputs and one for the hidden state. They are simply added together, so the second adds nothing to the model, but cuDNN requires it, which is why the results differ slightly.

---

Forecasting using a deep RNN. Stacking several layers of cells gives a deep RNN. In PyTorch, this only requires setting the RNN module's number of layers, for example to three, with the rest of the code unchanged. The three-layer model reaches a validation MAE of about twenty-nine thousand three hundred, the best so far.

---

Forecasting multivariate time series. Neural networks can handle multivariate time series with almost no change to their architecture. The book forecasts the rail ridership using both the rail and bus series as inputs, plus the day type. Whether tomorrow is a weekday, a weekend, or a holiday is known in advance, so the day type series is shifted one day into the future, and the model receives tomorrow's day type. It is one-hot encoded into three columns, giving five input columns in all.

The dataset class is extended so that each target keeps only its first value, the rail ridership, rather than all five columns. Input batches then have the shape thirty-two by fifty-six by five, and targets the shape thirty-two by one. The model is the same simple RNN, with an input size of five. It reaches a validation MAE of about twenty-three thousand two hundred, a big improvement.

Forecasting both rail and bus only requires keeping the first two values of each target and setting the output size to two. Using a single model for several related tasks often performs better than a separate model per task, because features learned for one task can help the others, and having to perform well on several tasks prevents overfitting, which is a form of regularization. It depends on the task, though. Here the multitask RNN does slightly worse than dedicated models, with validation MAEs of about twenty-six thousand four hundred for rail and twenty-six thousand two hundred for bus, which is still good.

---

Forecasting several time steps ahead. Predicting the value two weeks ahead only requires changing the targets to the values fourteen days ahead instead of one. Predicting all of the next fourteen values with a single model takes one of two approaches.

The first approach reuses the univariate RNN. It predicts the next value, appends it to the inputs as if it had actually occurred, predicts the following value, and so on, fourteen times. If the model makes an error at one step, the forecasts for all the following steps are affected, so errors tend to accumulate. This approach is best kept to a small number of steps.

The second approach trains a sequence-to-vector RNN to output all fourteen values in one shot. A new dataset class makes each target the next fourteen rail values, and the multivariate RNN gets an output size of fourteen. Its forecasts for the next day are better than its forecasts fourteen days out, but errors do not accumulate. The two approaches can also be combined: forecast fourteen days in one shot, append the forecasts to the inputs, and run the model again for the following fourteen days.

---

Forecasting using a sequence-to-sequence model. Instead of forecasting the next fourteen values only at the last time step, the model can be trained to forecast the next fourteen values at every time step. At time step zero it outputs forecasts for time steps one to fourteen, at time step one for time steps two to fifteen, and so on. The target for each input window is then a sequence of the same length as the input, with a fourteen-dimensional vector at each step, and the output has the shape batch size by window length by output size. This is a sequence-to-sequence, or seq2seq, RNN.

The targets contain values that also appear in the inputs, except at the last time step, but this is not cheating. At each time step, an RNN only knows about past time steps and cannot look ahead. It is a causal model.

After training, only the outputs at the last time step matter, and the rest can be ignored. The advantage is during training: the loss contains a term for the output at every time step, so many more error gradients flow through the model, and they do not have to flow as far through time. This can stabilize training and speed up convergence. And since the model must make predictions at every time step, it sees input sequences of varying lengths, which can reduce overfitting to the specific window length used in training.

The new dataset builds each target with PyTorch's unfold operation, which extracts sliding blocks, here windows of fourteen values shifted by one step. The model is the same RNN, except that the linear output layer is applied to the outputs at every time step instead of only the last. A linear layer accepts these three-dimensional outputs because PyTorch's matrix multiplication treats the leading dimensions as a grid of matrices and broadcasts the weight matrix across them. A one-dimensional convolutional layer with a kernel size of one would give the same result, with the time and feature dimensions swapped.

The seq2seq model reaches a validation MAE of about twenty-three thousand three hundred for the next day, which is very good. It is less accurate for more distant forecasts: the MAE for fourteen days ahead is about thirty-five thousand three hundred.

---

Handling long sequences. Simple RNNs do well on short sequences but not on long ones. To train on long sequences, an RNN must run over many time steps, so the unrolled network becomes very deep. Like any deep network, it may suffer from unstable gradients, which can make training take forever or become unstable. And while processing a long sequence, an RNN gradually forgets the first inputs.

---

Fighting the unstable gradients problem. Many of the tricks used for deep networks also help RNNs: good parameter initialization, faster optimizers, dropout, and so on. But nonsaturating activation functions, such as ReLU, may not help as much, and may even make an RNN more unstable. Suppose gradient descent updates the weights in a way that slightly increases the outputs at the first time step. The same weights are used at every time step, so the outputs at the second step may also increase slightly, then those at the third, until the outputs explode, and a nonsaturating activation function does not prevent this. A smaller learning rate reduces the risk, and so does a saturating activation function like tanh, which is why tanh is the default.

The gradients themselves can explode too. If training is unstable, it helps to monitor the size of the gradients and perhaps use gradient clipping.

Batch normalization cannot be used as efficiently with RNNs as with feedforward networks: it works between recurrent layers, but not between time steps. A batch norm layer can technically be added inside a memory cell, but then the same layer, with the same parameters, is used at every time step, regardless of the actual scale and offset of the inputs and hidden state. In 2015, César Laurent and colleagues showed that batch norm was slightly beneficial only when applied to a layer's inputs, not to the hidden states: between recurrent layers, but not within them.

Layer normalization tends to work a bit better than batch norm within recurrent layers. It is usually applied just before the activation function, at each time step. PyTorch's RNN module does not support it, so the recurrent loop must be written by hand, as in the first simple RNN, with a layer normalization module added to the memory cell just before tanh. This loses the simplicity and speed of the RNN module. Layer norm does not always help. It generally works better in gated RNNs, such as LSTM and GRU, and it is more likely to help when the series has been preprocessed to remove seasonality and trend.

Similarly, applying dropout between time steps requires a custom RNN, since the RNN module does not support it. The module does support dropout between recurrent layers, through its dropout hyperparameter, which defaults to zero. When forecasting, error bars are often useful, and Monte Carlo dropout is one way to get them.

---

Tackling the short-term memory problem. Data is transformed at every time step as it goes through an RNN, and some information is lost each time. After a while, the RNN's state contains virtually no trace of the first inputs. The book compares this to Dory, the fish with short-term memory loss from Finding Nemo, trying to translate a long sentence: by the time she has finished reading it, she has no clue how it started. Various cells with long-term memory solve this, and they have been so successful that basic cells are not used much anymore.

---

Long short-term memory cells. The long short-term memory cell, or LSTM, was proposed in 1997 by Sepp Hochreiter and Jürgen Schmidhuber, and gradually improved over the years by researchers such as Alex Graves, Haşim Sak, and Wojciech Zaremba. PyTorch's LSTM module is a drop-in replacement for the RNN module, and it usually performs much better: training converges faster, and the model detects longer-term patterns in the data.

From the outside, an LSTM cell looks like a regular cell, except that its state is split into two vectors: the short-term state, called h, and the long-term state, called c, for cell. The key idea is that the network can learn what to store in the long-term state, what to throw away, and what to read from it.

As the previous long-term state passes through the cell, it first goes through a forget gate, which drops some memories. Then new memories, selected by an input gate, are added to it. The result is sent straight out as the new long-term state, without any further transformation. So at each time step, some memories are dropped and some are added. After the addition, a copy of the long-term state also goes through tanh and is filtered by an output gate. This produces the new short-term state, which is also the cell's output for this time step.

The new memories and the gates come from four fully connected layers, each fed the current input vector and the previous short-term state. The main layer outputs a vector called g. It has the usual role of analyzing the current inputs and the previous short-term state, and in a simple RNN cell it is the only layer, with its output going straight out. In an LSTM cell, its output does not go straight out. Instead, its most important parts are stored in the long-term state, and the rest is dropped.

The other three layers are gate controllers. They use the logistic activation function, so their outputs range from zero to one, and they feed element-wise multiplications: outputs of zero close the gate, and outputs of one open it. The forget gate controls which parts of the long-term state are erased. The input gate controls which parts of g are added to the long-term state. The output gate controls which parts of the long-term state are read and output at this time step, both as the short-term state and as the output.

In short, an LSTM cell can learn to recognize an important input, which is the role of the input gate; store it in the long-term state; preserve it for as long as it is needed, which is the role of the forget gate; and extract it whenever it is needed, which is the role of the output gate; all while being fully differentiable. This is why LSTM cells have been so successful at capturing long-term patterns in time series, long texts, and audio recordings.

In equations, each gate controller's output is the logistic function applied to the sum of the current inputs times that gate's input weight matrix, the previous short-term state times its recurrent weight matrix, and its bias term. The main layer's output, g, is computed the same way but with tanh. The new long-term state is the forget gate's output times the previous long-term state, plus the input gate's output times g, with both products taken element by element. The new short-term state, which equals the output, is the output gate's output times tanh of the new long-term state, element by element. Each of the four layers has its own input weights, its own recurrent weights, and its own bias.

As with simple RNNs, adding layer normalization or dropout at each time step requires a manual recurrent loop. PyTorch's LSTM cell module runs a single time step, so the loop keeps two state tensors, the short-term state and the long-term state, both initialized to zeros, and passes them through the cell at each step.

---

Gated recurrent unit cells. The gated recurrent unit cell, or GRU, was proposed by Kyunghyun Cho and colleagues in 2014, in the paper that also introduced the encoder-decoder network. It is a simplified version of the LSTM cell, and it often performs just as well; one study suggests that all LSTM variants perform roughly the same.

The main simplifications are these. Both state vectors are merged into a single state vector. A single gate controller, called z, controls both the forget gate and the input gate. When it outputs one, the forget gate is open and the input gate closed; when it outputs zero, the opposite happens. In other words, whenever a memory must be stored, the location where it will be stored is erased first. There is no output gate: the full state vector is output at every time step. But a new gate controller, called r, controls which part of the previous state is shown to the main layer.

In equations, the outputs of z and r are each the logistic function of the current inputs and the previous state, each multiplied by its own weights, plus a bias. The main layer's output, g, is tanh of the inputs times their weights, plus the previous state, first multiplied element by element by r, times the recurrent weights, plus a bias. The new state is z times the previous state, plus one minus z times g, element by element.

PyTorch provides a GRU module, which replaces the RNN or LSTM module directly, and a GRU cell module for custom recurrent loops.

LSTM and GRU cells are one of the main reasons for the success of RNNs. Yet their short-term memory is still fairly limited, and they have a hard time learning long-term patterns in sequences of one hundred time steps or more, such as audio samples, long time series, or long sentences. One way to solve this is to shorten the input sequences, for example with one-dimensional convolutional layers.

---

Using one-dimensional convolutional layers to process sequences. A two-dimensional convolutional layer slides small kernels across an image and produces one two-dimensional feature map per kernel. Similarly, a one-dimensional convolutional layer slides several kernels across a sequence and produces one one-dimensional feature map per kernel. Each kernel learns to detect a single very short sequential pattern, no longer than the kernel size. With ten kernels, the output is ten sequences of the same length, or equivalently a single ten-dimensional sequence. So a network can mix recurrent layers with one-dimensional convolutional layers, or even one-dimensional pooling layers.

PyTorch's one-dimensional convolution layer expects inputs shaped batch size by input features by sequence length, and produces outputs shaped batch size by output features by sequence length, so the last two dimensions must be swapped before and after it. With a stride of one and same padding, the output sequence has the same length as the input. With valid padding or a stride greater than one, it is shorter, so the targets must be adjusted.

The book's downsampling model has a one-dimensional convolutional layer with a kernel size of four and a stride of two, then ReLU, a GRU layer, and a linear output layer, all working on whole sequences, and it ignores the hidden states returned by the GRU layer. The stride halves the length of the sequences. Since the kernel is larger than the stride, every input is used to compute the layer's output, so the model can learn to keep the most useful information and drop only unimportant details.

Shorter sequences help the GRU layer detect longer patterns, so the window length can be doubled to one hundred twelve days. The targets must match the convolution's output. The kernel size is four, so the first output of the convolutional layer is based on input time steps zero to three, and the first forecasts must be for time steps four to seventeen instead of one to fourteen. So the first three target steps are cropped off, and the targets are downsampled by a factor of two to match the stride.

---

WaveNet. In a 2016 paper, Aaron van den Oord and other DeepMind researchers introduced WaveNet, which uses only one-dimensional convolutional layers and no recurrent layers. It stacks convolutional layers and doubles the dilation rate, meaning how spread apart each neuron's inputs are, at every layer. The first layer sees two time steps at a time, the next one sees four, the next eight, and so on. The lower layers learn short-term patterns, the higher layers learn long-term ones, and thanks to the doubling dilation rate, the network can process extremely long sequences very efficiently.

The paper's authors stacked ten convolutional layers with dilation rates of one, two, four, eight, and so on up to five hundred twelve, then a second identical group of ten layers, then a third. A single group of ten such layers acts like a convolutional layer with a kernel of one thousand twenty-four, except that it is much faster, more powerful, and uses far fewer parameters. Before every layer, the input sequences are padded on the left with as many zeros as the dilation rate, so the sequence length stays the same throughout the network. Padding only on the left, rather than on both sides, ensures the convolutional layers never peek into the future when making predictions, which makes WaveNet a causal model.

The book implements a simplified WaveNet. A custom causal convolution module behaves like PyTorch's one-dimensional convolution layer, except that it first pads its input on the left by the kernel size minus one, times the dilation rate, so the sequence keeps its length. The model stacks eight of these layers, each with a kernel size of two, with dilation rates of one, two, four, and eight, then one, two, four, and eight again, each followed by ReLU, and ends with a linear output layer. The time and feature dimensions are swapped before and after the convolutions. Thanks to the causal padding, every layer outputs a sequence as long as its input, so the full target sequences can be used without cropping or downsampling, and the data loaders from the seq2seq model can be reused.

The complete WaveNet uses a few more tricks, such as skip connections like those in a ResNet, and gated activation units similar to those in a GRU cell.

These sequence models perform similarly on the ridership task, but they can differ a lot depending on the task and the amount of data. In the WaveNet paper, the authors achieved state-of-the-art performance on various audio tasks, hence the name, including text-to-speech with very realistic voices across several languages, and they generated music one audio sample at a time. A single second of audio can contain tens of thousands of time steps, too many even for LSTMs and GRUs.

---

A caveat about the test period. The best ridership models perform much worse than expected on the test period, which runs into 2020, when the Covid-19 pandemic greatly affected public transportation. These models only work well if the patterns they learned from the past continue into the future. A model should be checked on recent data before it is deployed, and its performance should be monitored regularly once it is in production.

---

Summary.

Recurrent neural networks have connections pointing backward, so their output at each time step depends on the current inputs and on a state carried over from the previous step. They handle sequences of any length, in sequence-to-sequence, sequence-to-vector, vector-to-sequence, and encoder-decoder configurations, and they are trained by unrolling them through time and using regular backpropagation.

For time series, naive forecasting and the ARMA family, including ARIMA and SARIMA, are useful baselines. Differencing removes trend and seasonality to make a series more stationary. Machine learning models are trained on every past window of the series, with the following values as targets.

On the Chicago ridership data, a simple RNN beat both baselines, a deep RNN did slightly better, and adding the bus series and the next day's type helped most. Several future steps can be forecast one at a time, all at once, or at every time step with a sequence-to-sequence model, which provides more gradients and more stable training.

Long sequences cause unstable gradients and short memory. Saturating activations, smaller learning rates, gradient clipping, layer normalization, and dropout address the first. LSTM and GRU cells, whose gates learn what to store, keep, and read, address the second. One-dimensional convolutional layers can shorten sequences before a recurrent layer, and WaveNet's stacked causal convolutions with doubling dilation rates handle sequences of tens of thousands of time steps. Every forecasting model relies on past patterns continuing, so it must be checked on recent data and monitored in production.
