# Chapter 10 — Building Neural Networks with PyTorch, to fall asleep to

Chapter ten. Building neural networks with PyTorch.

---

PyTorch. PyTorch is an open source deep learning library developed by Facebook's AI Research lab, now Meta AI. It is the Python successor of the Torch library, which was written in Lua. It resembles NumPy, but adds hardware acceleration on graphics processing units, or GPUs, and other accelerators, plus automatic differentiation, optimizers, and ready-to-use neural network components.

When PyTorch was released in 2016, Google's TensorFlow was by far the most popular library: fast and scalable, but with a complex, static programming model that was hard to use and debug. PyTorch was designed to be flexible and Pythonic. It uses dynamic computation graphs, also called define-by-run, which makes it intuitive and easy to debug. It became the most used framework in academia, and much of the industry followed. In 2022 its governance moved to the PyTorch Foundation, part of the Linux Foundation. TensorFlow version two and Google's JAX remain strong competitors, and the libraries have converged enough that switching between them is much easier than it used to be.

---

Tensors. The core data structure of PyTorch is the tensor: a multidimensional array with a shape and a data type, used for numerical computation. It is much like a NumPy array, with two extra features: it can live on a GPU, and it supports automatic differentiation. Neural networks in PyTorch take tensors as inputs and return tensors as outputs, the way Scikit-Learn models work with NumPy arrays.

A tensor is created much like a NumPy array, for example from a nested list of numbers. It holds floats, integers, booleans, or complex numbers, with one data type per tensor. With mixed initial values, the most general type wins: complex, then float, then integer, then boolean. Tensors of strings or objects are not supported. Indexing and slicing work as in NumPy, and the math functions mirror NumPy's: absolute value, cosine, exponential, maximum, mean, square root, and so on. Most also exist as tensor methods. Arithmetic operators act itemwise, and there is an operator for matrix multiplication. Reductions take an argument named dim, short for dimension, though axis is also accepted. Taking the maximum along dimension zero returns both the maximum of each column and the indices where they occur.

Tensors convert to NumPy arrays and back. PyTorch's default float precision is thirty-two bits, while NumPy's is sixty-four. Thirty-two bits is generally better for deep learning: it takes half the memory, runs faster, and neural networks do not need the extra precision. So when converting a NumPy array, thirty-two-bit floats should be requested explicitly, or the float tensor constructor used, which converts automatically. Both copy the data. The from numpy function instead shares the array's memory, so modifying one modifies the other.

PyTorch has many in-place operations, which modify a tensor directly and can save memory and time. The in-place ReLU, for example, replaces every negative value with zero. In-place operations are easy to spot, because their names end with an underscore. With very few exceptions, such as the zero operation, removing the underscore gives the regular operation.

---

Hardware acceleration. Tensors can be copied to a GPU if the machine has a compatible one and the right libraries. A Colab GPU runtime, such as one with an Nvidia T4, comes with PyTorch compiled for GPUs, plus Nvidia's CUDA platform and cuDNN, a CUDA library that accelerates deep neural networks. Besides Nvidia GPUs, PyTorch supports Apple's Metal Performance Shaders, or MPS, on Apple silicon; AMD GPUs through the ROCm software stack; Intel GPUs and central processing units, or CPUs, through Intel's oneAPI; and Google's tensor processing units, or TPUs, through the torch XLA library, XLA being short for Accelerated Linear Algebra.

The usual pattern checks whether CUDA is available, then whether MPS is available, and otherwise falls back to the CPU, and stores the result as the device to use. Deep learning needs a lot of compute, so a hardware accelerator matters. Colab and Kaggle offer free GPUs.

A tensor can be created on the CPU and then copied to the GPU with its to method, or created directly on the GPU with a device argument. Its device attribute shows where it lives. Multiple Nvidia GPUs are addressed by index: cuda zero, cuda one, and so on. Operations on GPU tensors run on the GPU, and their results stay there, so many operations can run without moving data between the CPU and the GPU. This matters, because transfers between devices are often the performance bottleneck.

The speedup depends on the GPU, since expensive ones are dozens of times faster than cheap ones, and on data throughput. For a compute-heavy model, the GPU's speed and memory matter most. For a shallow model, feeding it the data may be the bottleneck. In the book's test, multiplying a thousand-by-thousand matrix by its transpose ran about twenty-nine times faster on Colab's free GPU than on the CPU, but for hundred-by-hundred matrices it was only about twice as fast. GPUs split large operations into many small ones and run them in parallel across thousands of cores. Small tasks cannot be split that much, and for many tiny tasks the CPU can even be faster.

---

Autograd. PyTorch's reverse-mode automatic differentiation is called autograd, short for automatic gradients. Take the function f of x equals x squared. Its derivative is two x, so at x equals five the function equals twenty-five and its derivative equals ten.

In PyTorch, x is created as a tensor equal to five, with the requires grad argument set to true. This marks x as a variable rather than a constant, so PyTorch tracks every operation involving it and captures the computation graph that backpropagation needs. In that graph, x is a leaf node. Computing x squared gives a tensor equal to twenty-five, which carries a grad function attribute: the operation that created it, here the power operation, which tells PyTorch how to backpropagate gradients through it. Calling backward on the result backpropagates the gradients back to the leaf nodes, and the grad attribute of x then holds ten.

PyTorch builds a new computation graph on the fly during each forward pass, which supports very dynamic models with loops and conditionals.

Gradients accumulate. In TensorFlow or JAX, computing gradients just returns them. In PyTorch, calling backward adds the gradients into the grad attribute of every variable used. Calling backward on two tensors that share a variable leaves that variable holding the sum of both gradients.

A gradient descent step subtracts a fraction of the gradients from the variables. The step itself must not be tracked, and PyTorch raises an exception for an in-place operation on a tracked variable, so the step runs inside a no grad context, which temporarily disables gradient tracking. With a learning rate of zero point one, x goes from five to four. Another option is the detach method, which creates a new tensor outside the computation graph that shares the same memory. Detach is handy for evaluation, logging, or fine-grained control over which operations contribute to gradients, while the no grad context is generally preferred for inference and gradient descent steps.

Before the next iteration, the gradients must be reset by calling the in-place zero method on each grad tensor. Forgetting this makes gradients accumulate, which gives wrong updates. There is no error message, just poor performance and perhaps infinite or not-a-number values, so the bug is hard to find.

A complete training loop therefore repeats four things: a forward pass, a backward pass, a gradient descent step inside a no grad context, and a reset of the gradients.

---

In-place operations and autograd. In-place operations do not always work with autograd. First, an in-place operation on a leaf node that requires gradients raises a runtime error, because PyTorch would not know where to store the computation graph.

Second, an in-place operation can destroy a value the backward pass needs. If z is the exponential of t, and one is then added to z in place, z is correct, but calling backward raises a runtime error, because the exponential's output was overwritten. Assigning z plus one to z instead works: a new tensor is created, and the original stays in the graph.

Operations differ in what they save. Some save their outputs, such as exponential, ReLU, sigmoid, square root, and hyperbolic tangent, so their outputs must not be modified in place. The exponential saves its output because its derivative equals its output. Others save their inputs, such as absolute value, cosine, log, and sine, so their inputs must not be modified in place before the backward pass. Absolute value must keep its input, because its output never reveals the input's sign. Some save both, such as maximum, minimum, product, and standard deviation. A few save neither, such as floor, mean, round, and sum, and can be modified freely. That is why replacing the exponential with the cosine in the earlier example produces no error.

The recommended practice is to write models without in-place operations first, then convert the most costly operations if memory or speed requires it, checking that the outputs stay the same and that no tensor needed for backpropagation is modified in place.

---

Linear regression using tensors and autograd. The example uses the California housing dataset from chapter nine, split into training, validation, and test sets. The features are converted to thirty-two-bit float tensors and standardized with tensor operations, using the mean and standard deviation of each feature computed on the training set. The targets are reshaped into column vectors, with a second dimension of size one, to match the predictions. Column vectors avoid ambiguity in matrix multiplication and broadcasting.

The parameters are a weight column vector, one weight per feature, eight here, initialized randomly, and a scalar bias initialized to zero, both with gradient tracking on. Zero weights would also work for linear regression, but neural networks need random initialization to break the symmetry between neurons, so it is a good habit.

A random seed is set for reproducibility, but PyTorch does not guarantee identical results across releases, platforms, or devices. A GPU runs chunks of each operation in parallel, and the order they finish in can vary, causing tiny floating-point differences that can compound during training. PyTorch can be restricted to deterministic algorithms, but these are often slower, and some operations have no deterministic version.

Training uses batch gradient descent, with the full training set at every step, a learning rate of zero point four, and twenty epochs. Each epoch computes the predictions as the inputs times the weights plus the bias, then the mean squared error loss. Calling backward on the loss computes the gradients for the weights and the bias. Both are updated inside a no grad context, and both gradients are reset to zero. The item method extracts a plain number from a scalar tensor for printing. The loss falls from about sixteen to about zero point five seven. Predictions on new data are also made inside a no grad context, which uses less memory and runs faster, since no graph is recorded.

---

Linear regression using the high-level API. The low-level approach gets messy for complex models, so PyTorch provides a higher-level application programming interface, or API, in its neural network package, conventionally imported as nn. Its linear module implements linear regression. Every module subclasses PyTorch's base module class, and most networks are assembled from many modules, like building bricks.

A linear module holds a bias vector, one term per neuron, and a weight matrix with one row per neuron and one column per input, the transpose of the hand-built version. Both are initialized randomly, from a uniform distribution between minus and plus one over the square root of the number of inputs. They are instances of the parameter class, a subclass of tensor. A module's parameters method returns all of its parameters, and recursively those of its submodules. It does not return regular tensors, even ones that require gradients, and that is the main difference between a tensor and a parameter. The named parameters method also returns their names.

A module is called like a function, which runs its forward method. The linear module's forward method multiplies the inputs by the transposed weight matrix and adds the bias. Forward hooks registered on a module run every time it is called, which helps with logging and debugging. Hooks only run when the module is called like a function, so its forward method should never be called directly.

Training needs an optimizer and a loss function. The stochastic gradient descent optimizer, or SGD, created from the model's parameters and a learning rate, handles stochastic, mini-batch, and batch gradient descent. The loss is an instance of the mean squared error loss module, called on the predictions and the targets. In PyTorch, the loss function object is usually called the criterion, to distinguish it from the loss value it computes at each iteration.

The training loop now uses these tools. For each epoch, it calls the model on the inputs, computes the loss with the criterion, and calls backward on the loss. The optimizer's step method then updates the parameters, and its zero grad method resets the gradients. The optimizer handles the no grad context itself. Many people call zero grad before backward rather than after, which is slightly safer. The loss reaches about zero point five four after twenty epochs.

---

A regression MLP. The sequential module chains modules: it feeds its input to the first module, that module's output to the second, and so on. Many neural networks are just one stack of modules, which makes it one of the most useful modules in PyTorch.

The regression multilayer perceptron, or MLP, is a sequential module with a linear layer from the eight features to fifty outputs, a ReLU, a linear layer from fifty to forty, another ReLU, and a linear output layer from forty to one. The first layer's inputs must match the number of features, but its number of outputs is a hyperparameter. The ReLU module has no parameters and keeps its input's shape. Each layer's inputs must match the previous layer's outputs. The output layer's size must match the targets' dimensionality, here one. With SGD and a learning rate of zero point one, the loss reaches about zero point five seven. This is still batch gradient descent, which does not scale to large datasets.

---

Mini-batch gradient descent with data loaders. PyTorch's data loader loads batches of a chosen size efficiently and can shuffle the data at each epoch. It expects a dataset object with two methods: one returning the number of samples, and one returning the sample and its target at a given index. The tensor dataset class wraps existing tensors in this interface. The training set is wrapped in a tensor dataset, and a data loader serves shuffled mini-batches of thirty-two.

To use the GPU, the model's to method moves all its parameters to GPU memory, and each batch is copied to the GPU at the start of each iteration. The optimizer must be created after the model is moved, because some optimizers keep internal state on the same device as the parameters.

The training function first calls the model's train method. Then, for each epoch, it loops over the data loader. For each batch, it moves the inputs and targets to the device, computes the predictions and the loss, adds the loss to a running total, calls backward, calls the optimizer's step method, and calls zero grad. At the end of each epoch, it reports the mean loss.

The train method switches the model and its submodules to training mode. This matters for layers that behave differently in training and evaluation, such as dropout and batch normalization, covered in chapter eleven. Before evaluation or prediction, the model must be switched to evaluation mode with its eval method.

PyTorch has no built-in training loop; it is written by hand, which gives freedom and control. Libraries built on PyTorch, such as PyTorch Lightning, FastAI, Catalyst, and Keras, provide tested training loops with extras such as multi-GPU support.

With mini-batches and a learning rate of zero point zero two, the loss reaches about zero point three two in twenty epochs, much lower than before, but each epoch is slower. Two tweaks help. First, on a CUDA device, the data loader's pin memory option allocates the data in page-locked memory, at a fixed physical location, which allows direct memory access transfers to the GPU and removes an extra copy. It can use more memory, since that memory cannot be swapped to disk, but transfers are usually much faster. Tensors can then also be moved to the GPU with the non-blocking option, so the CPU is not blocked. Second, setting the data loader's number of workers makes separate processes load and preprocess upcoming batches while the GPU works on the current one. The best number must be found by experiment, and when spawning workers slows things down, especially on Windows, persistent workers can be reused across epochs.

---

Model evaluation. An evaluation function takes the model, a data loader, a function computing a metric for one batch, and a function aggregating the batch metrics, by default the mean. It switches the model to evaluation mode, and inside a no grad context runs every batch through the model, computes the batch metrics, stacks them, and aggregates them. The validation mean squared error, or MSE, is about zero point four one.

Using the root mean squared error, or RMSE, as the batch metric gives about zero point five seven, but the square root of the validation MSE is about zero point six four. The mean of the per-batch RMSEs is not the RMSE of the whole set. The fix is to use the MSE as the batch metric and make the aggregation function take the square root of the mean. Averaging batch MSEs gives the overall MSE because the batches are the same size, apart from a usually smaller last batch, which makes very little difference.

The TorchMetrics library, from the team behind PyTorch Lightning, provides well-tested streaming metrics. A streaming metric tracks a metric and is updated one batch at a time. The evaluation function resets it at the start, calls its update method on each batch's predictions and targets, and calls its compute method at the end. Its mean squared error metric, with squaring turned off and moved to the GPU, gives the correct RMSE of about zero point six four.

The training function can also report training performance during each epoch and validation performance at the end of each epoch. Training performance much better than validation performance suggests overfitting, or a bug such as a data mismatch between the two sets. Learning curves, plotted with Matplotlib or TensorBoard, make this easier to see.

---

Nonsequential models with custom modules. A Wide and Deep neural network, introduced by Heng-Tze Cheng and colleagues in 2016, connects all or part of the inputs directly to the output layer. It can learn deep patterns through the deep path and simple rules through the short path, which can also carry manually engineered features. A regular MLP forces all data through the full stack of layers, which can distort simple patterns.

A nonsequential architecture needs a custom module: a class derived from the base module class, whose constructor calls the base constructor and creates the layers, and whose forward method defines how they are used. The Wide and Deep module's constructor builds the deep part as a sequential module with hidden layers of fifty and forty neurons and ReLU activations, and a separate output layer taking forty plus the number of features as inputs. Its forward method runs the input through the deep stack, concatenates the original input with the deep stack's output along dimension one, and feeds the result to the output layer.

A model with a variable number of submodules should store them in a module list or module dictionary, not a regular Python list or dictionary, so PyTorch registers them. A variable number of parameters likewise goes in a parameter list or parameter dictionary.

The custom module is trained, evaluated, and used exactly like the earlier models, though the best learning rate changes, here to about zero point zero zero two. To send one subset of features through the wide path and a different, possibly overlapping subset through the deep path, the forward method can slice the input.

---

Multiple inputs. Some models need inputs that cannot easily be combined into one tensor, such as images and text. The forward method can take a separate wide input and deep input. The tensor dataset then wraps three tensors, the wide features, the deep features, and the targets, and the training and evaluation loops move each to the device and pass both inputs to the model. Python's unpacking operator can pass all the inputs in order, which works for any number of inputs, as long as their order matches the forward method's arguments.

With many inputs, mixing up the order causes hard-to-debug errors. Naming the inputs avoids this. A custom dataset class, derived from PyTorch's dataset class, returns for each index a dictionary mapping input names to values, along with the target. When the dictionary's keys match the forward method's argument names, Python's double-star unpacking passes them all as keyword arguments.

---

Multiple outputs. Several situations call for multiple outputs. The task may need them, such as locating and classifying the main object in a picture, which is both regression and classification. There may be several independent tasks on the same data, and one network with one output per task often beats separate networks, because it learns features useful across tasks: for example, one output for a face's expression and another for whether the person wears glasses. Another use is regularization, meaning a training constraint that reduces overfitting. An auxiliary output can ensure that part of the network learns something useful on its own.

An auxiliary output added to the Wide and Deep model makes sure the deep part can predict well alone. A new linear layer maps the deep stack's forty outputs down to one, and the forward method returns both the main and the auxiliary output. During training, the criterion is applied to each output, and the losses are combined into a single loss, generally a weighted sum: here, zero point eight times the main loss plus zero point two times the auxiliary loss. The main loss weighs more because it matters most, and the ratio is a regularization hyperparameter. When outputs need different targets, a custom dataset returns them all. During evaluation, the auxiliary output is ignored.

---

Loading Fashion MNIST with TorchVision. TorchVision is an important part of the PyTorch ecosystem. It provides computer vision tools: functions to download common datasets such as Fashion MNIST, pretrained models, image transforms such as crop, rotate, and resize, and more. Fashion MNIST comes split into sixty thousand training images and ten thousand test images. PyTorch's random split function holds out five thousand of the training images for validation.

By default, the dataset returns images in the format of the Python Imaging Library, or PIL, with integer pixel values from zero to two hundred fifty-five, but the network needs float tensors with scaled values. TorchVision datasets accept a transform argument: a preprocessing function run on the fly whenever data is accessed. The transform here chains two steps: one converts the image into TorchVision's image class, a tensor subclass, and the other converts it to thirty-two-bit floats, scaled between zero and one. Version two of TorchVision's transforms API should be used, because it is faster and has more features. Data loaders with batches of thirty-two are created for each set, shuffling only the training set.

Each image is now a tensor of shape one by twenty-eight by twenty-eight, instead of chapter nine's flat array of seven hundred eighty-four pixels. The first dimension is the channel dimension: grayscale images have one channel, and color images usually have three. The other two are height and width. PyTorch expects channels first, while Matplotlib, PIL, TensorFlow, and OpenCV expect them last, so the channel dimension must be moved for each library; the permute function reorders dimensions. The targets are integers from zero to nine, and the dataset's classes attribute gives the class names. The first training image is an ankle boot.

---

Building the classifier. The classifier is a custom module wrapping a sequential stack: a flatten layer, a linear layer from seven hundred eighty-four inputs to three hundred neurons, a ReLU, a linear layer to one hundred neurons, a ReLU, and a linear output layer with ten outputs, one per class. A plain sequential module would work, but wrapping models in custom modules makes them easier to deploy, reuse, and tune. The flatten layer has no parameters; it reshapes each sample into one dimension, so a batch of shape thirty-two by one by twenty-eight by twenty-eight becomes thirty-two by seven hundred eighty-four, as linear layers require. The output layer has no activation function.

The loss is the cross-entropy loss, which accepts either class indices or class probabilities as targets. Chapter nine used softmax on the output layer, but PyTorch's cross-entropy loss works directly from the logits, the raw class scores. This skips logarithms and exponentials that cancel out, saving compute and memory, and is more numerically stable. The downside is that softmax must be applied manually to get class probabilities.

---

Other classification losses. For multiclass classification, an alternative is a log softmax activation on the output layer with the negative log-likelihood loss. The model then outputs log probabilities, and the exponential function turns them into probabilities. It is a bit slower, so less common, but useful when log probabilities are wanted, or when the distribution needs adjusting before the loss.

For binary classification, the output layer has one neuron, and the loss is the binary cross-entropy with logits loss. The model outputs logits, and the sigmoid function gives the probability of the positive class. A sigmoid activation with the plain binary cross-entropy loss outputs probabilities directly, but is a bit slower and less numerically stable. For multilabel binary classification, the output layer has one neuron per label.

---

Training and using the classifier. The classifier is trained with the same training function and SGD, and evaluated with the TorchMetrics accuracy metric for ten classes. Training takes a few minutes on a GPU, much longer without one. The model reaches about ninety-two point eight percent accuracy on the training set and eighty-seven point two percent on the validation set, a little overfitting, which fewer neurons or regularization could reduce.

For predictions, the model is switched to evaluation mode and run on new images inside a no grad context. The predicted class is the one with the highest logit. The first three validation images are predicted as a sneaker, a coat, and a pullover, all correct. Applying softmax to the logits gives the estimated probabilities, here ninety-one point one percent confidence for the first image and ninety-nine point six percent for the second. The softmax function from PyTorch's functional module avoids creating a module just for this. The top k function returns the k largest logits and their indices: the first image's second most likely class is ankle boot, at eight point eight percent. Label smoothing is set through the cross-entropy loss's label smoothing hyperparameter, for example zero point zero five.

Fashion MNIST is balanced. For an unbalanced dataset, rare classes should get more weight and frequent classes less, or the model will be biased toward the frequent ones. The cross-entropy loss's weight argument does this. For classes of nine hundred, seven hundred, and four hundred instances, two thousand in total, each class weight is two thousand divided by its count, preferably normalized to sum to one, which gives about zero point two two, zero point two eight, and zero point five.

---

Fine-tuning hyperparameters with Optuna. A PyTorch model can be wrapped as a Scikit-Learn estimator, with a custom class or the Skorch library, and tuned with grid search or randomized search, but dedicated tuning libraries such as Optuna, Ray Tune, and Hyperopt usually do better.

Optuna calls an objective function once per trial. The function receives a trial object, asks it for hyperparameter values, builds and trains a model with them, evaluates it on the validation set, and returns the metric. The example tunes the learning rate and the number of neurons per hidden layer. The trial's suggest float method requests a learning rate between ten to the minus five and ten to the minus one, sampled on a log scale. Because the right scale is unknown, log sampling explores all scales, while a uniform distribution would almost never try tiny values. The suggest int method requests between twenty and three hundred neurons, and a suggest categorical method also exists.

Tuning starts by creating a study and calling its optimize method with the objective and a number of trials. Optuna minimizes by default, so the direction is set to maximize, since the objective returns accuracy. For reproducibility, both PyTorch's seed and the sampler's seed are set.

Optuna's default sampler is the Tree-structured Parzen Estimator, or TPE, a sequential model-based optimization algorithm. It starts with random values, learns from past results, and progressively focuses on the most promising regions of the hyperparameter space, so it finds much better hyperparameters than random search in the same time. More hyperparameters can be searched, such as batch size, optimizer, number of layers, or activation function, but the search space grows exponentially with each one.

After five trials, the best result was a learning rate of about zero point zero eight five and one hundred sixteen neurons, with about eighty-eight point seven percent validation accuracy. Fifty or more trials do much better but take hours. Optuna can also run trials in parallel on several machines, sharing results through a database such as SQLite or PostgreSQL, for a nearly linear speedup and a full log of all experiments. Rather than using global variables, the data loaders are better passed to the objective as extra arguments, bound with a lambda or a partial function.

---

Pruning trials. Some trials are clearly bad early, for example when the loss shoots up in the first epoch. Stopping them saves time and compute. The objective can return the current validation accuracy, or it can raise Optuna's trial pruned exception, which makes Optuna ignore the trial entirely. That often makes the search more efficient, because it keeps noisy evaluations out of the search algorithm.

Optuna's pruners detect bad trials. The median pruner stops trials performing below the median of past trials at the same epoch. It starts after a number of startup trials have completed, five by default, and in each later trial it waits a number of warmup steps, then checks at regular intervals. In the objective, after each epoch, the trial's report method sends the validation accuracy and epoch to Optuna, and if the trial's should prune method returns true, the objective raises the trial pruned exception.

Once the hyperparameters are chosen, the model is trained on the training set plus the validation set, then evaluated on the test set.

---

Saving and loading models. The simplest way to save a model is PyTorch's save function, given the model and a file path. The model is serialized with Python's pickle module, compressed, and written to disk, conventionally with a pt or pth extension. The load function, with its weights only argument set to false, restores the whole model object. Custom classes, such as the image classifier class, are saved only as references, so their code must be loaded first, in the same version. A loaded model must be switched to evaluation mode before inference.

This approach has serious drawbacks. Pickle is notoriously insecure: an attacker could inject malicious code into a saved model, and pickle would run it on loading, so only fully trusted models should be loaded this way. Pickle is also brittle: it varies across Python versions, and it stores file paths to locate code, which break if the folder structure changes.

The recommended approach is to save only the weights, through the model's state dictionary: an ordered dictionary with one entry per named parameter, plus any buffers. A buffer is a regular tensor registered with a module to hold extra data that is not a parameter; batch normalization layers, in chapter eleven, use buffers. To load, a model with exactly the same architecture is created, the file is loaded with weights only set to true, and the weights go into the model's load state dictionary method. The file contains only data, which is safe and much less likely to break across Python versions or in deployment.

Rebuilding the architecture requires knowing the number of layers, the neurons per layer, and so on, so it helps to save the model's hyperparameters alongside the state dictionary. Resuming training later also requires saving the optimizer's state dictionary, the current epoch, and the loss history. Hugging Face's safetensors library is another popular way to save weights safely.

---

TorchScript. PyTorch can convert a model's code to TorchScript, a statically typed subset of Python. TorchScript can be compiled into significantly faster models: several operations can be fused into one, operations on constants can be replaced by their results, which is called constant folding, and unused code can be pruned. TorchScript can also be saved and then run in Python, or in C plus plus through the LibTorch library, which brings PyTorch models to many devices, including embedded ones.

There are two ways to convert a model. Tracing runs the model on sample data, logs every operation, and converts the log. It works well for static models without conditionals or loops. With an if statement, only the branch actually executed is captured, and with a loop, the operations are copied once per iteration that ran.

Scripting parses the Python code directly. It handles if statements and while loops properly, as long as the conditions are tensors, and for loops over tensors. It supports only a subset of Python: no global variables, generators, complex list comprehensions, variable-length arguments, or match statements; types must be fixed; and any function called must follow the same rules, which rules out the standard library and third-party libraries. Most real-world models can respect these rules.

Either way, the TorchScript model can be further optimized for inference. TorchScript models can only run inference, not training, because they do not support gradient tracking or parameter updates. TorchScript is no longer under active development: bugs are fixed, but no features are added. It remains one of the best ways to run PyTorch models in C plus plus. Another option is exporting to the open ONNX standard, short for Open Neural Network Exchange, which runs in many environments.

---

Compiling models. Since PyTorch two point zero, released in March 2023, the PyTorch team has focused on compilation tools centered on the compile function. The compiled model is used normally and is compiled and optimized automatically when it runs. This is just-in-time compilation, as opposed to ahead-of-time compilation. Under the hood, TorchDynamo hooks into Python bytecode to capture the model's computation graph. Working at the bytecode level makes the capture efficient and reliable, handles conditionals and loops properly, and provides dynamic information for better optimization. A backend called TorchInductor then compiles the graph, using the Triton language to generate fast code for Nvidia GPUs, or the OpenMP interface for CPUs. Other backends exist, including XLA for Google's TPUs.

---

Summary.

PyTorch is a deep learning library built around tensors: multidimensional arrays like NumPy's that can live on a GPU and support automatic differentiation. Its default float precision is thirty-two bits, and transfers between the CPU and the GPU are often the bottleneck.

Autograd builds a computation graph on the fly during each forward pass. Calling backward accumulates gradients in each variable's grad attribute, so gradients must be reset at every iteration, and updates run inside a no grad context. In-place operations save memory but must not overwrite values the backward pass needs.

Models are built from modules. The linear module holds a weight matrix and a bias vector, the sequential module chains modules, and custom modules define a forward method, which allows any architecture, including multiple inputs and outputs.

The training loop is written by hand. A data loader serves shuffled mini-batches. For each batch, the data moves to the device, the model makes predictions, the criterion computes the loss, calling backward on the loss computes the gradients, the optimizer's step method updates the parameters, and zero grad resets the gradients. The model is put in training mode before training and evaluation mode before evaluation or inference.

For multiclass classification, the model outputs logits, and the cross-entropy loss works on them directly; softmax turns logits into probabilities. Binary classification uses one output and the binary cross-entropy with logits loss. TorchMetrics provides streaming metrics.

Optuna tunes hyperparameters with the Tree-structured Parzen Estimator and can prune bad trials. Models are best saved as state dictionaries rather than pickled objects. TorchScript and the compile function make models faster.
