# Appendix B — Mixed Precision and Quantization, to fall asleep to

Appendix B. Mixed precision and quantization.

---

Model size.

By default, PyTorch represents model parameters as thirty-two-bit floats, which take four bytes each. A model with one billion parameters needs at least four gigabytes of RAM just to hold its parameters. Inference also needs RAM for the activations, and training needs RAM for all the intermediate activations and for the optimizer's state: Adam keeps two extra values per parameter, another eight gigabytes. Large models also cost transfer time between CPU and GPU, storage, download time, and energy.

The simplest way to shrink a model is reduced precision, typically sixteen-bit floats instead of thirty-two-bit floats. Converting a trained thirty-two-bit model to sixteen bits halves its size, with little impact on quality. Training directly in sixteen bits can cause convergence problems, so a common strategy is mixed-precision training, or MPT: weights and weight updates stay in thirty-two-bit precision during training, the rest of the computation uses sixteen bits, and the weights are shrunk to sixteen bits after training.

To shrink a model further, quantization discretizes the parameters and represents them as eight-bit integers, or four bits or fewer. It is harder and degrades quality a bit more, but it reduces model size by a factor of four or more and speeds the model up significantly.

---

Floating-point formats.

PyTorch's thirty-two-bit floats follow the IEEE standard for floating-point arithmetic, IEEE 754, which specifies how floating-point numbers are stored in memory. The format can represent tiny and huge values, as well as special values: plus and minus zero, plus and minus infinity, and NaN, meaning not a number.

The float32 data type, or fp32, holds magnitudes from about one point four times ten to the minus forty-five up to about three point four times ten to the thirty-eight. Its first bit is the sign: zero for positive, one for negative. The next eight bits hold the exponent, from zero to two hundred fifty-five. The last twenty-three bits hold the fraction, from zero to two to the twenty-third minus one.

When the exponent is between one and two hundred fifty-four, the number is normalized, which is the most common case. Its value is plus or minus one, depending on the sign, times two to the power of the exponent minus one hundred twenty-seven, times one plus the fraction divided by two to the twenty-third. That last factor holds the most significant digits and is called the significand.

When the exponent is zero and the fraction is positive, the number is subnormal, used for the tiniest values: plus or minus the fraction times two to the minus one hundred forty-nine. An exponent and fraction both equal to zero give plus or minus zero. The maximum exponent, two hundred fifty-five, gives infinity when the fraction is zero, and NaN otherwise.

Other floating-point formats differ only in how many bits go to the exponent and to the fraction. Float16 uses five exponent bits, from zero to thirty-one, and ten fraction bits, from zero to one thousand twenty-three. A normalized float16 value is plus or minus two to the power of the exponent minus fifteen, times one plus the fraction divided by two to the tenth. Float8 uses four exponent bits and three fraction bits, so it is often written fp8 E4M3. The M stands for mantissa, an older and ambiguous term for the fraction.

The bfloat16 and bfloat8 formats were proposed by Google Brain, hence the b. They offer a wider range of values at the cost of significantly reduced precision.

---

Integer formats.

Integers commonly use sixty-four, thirty-two, sixteen, or eight bits. Quantization typically uses eight-bit integers, unsigned or signed.

Below eight bits, several weights are packed into each byte: two weights per byte for four-bit quantization, and four per byte for two-bit quantization. Weights can even be quantized to ternary values, minus one, zero, or plus one, usually stored five per byte. Three to the power five is two hundred forty-three, which is less than two hundred fifty-six, so five ternary values fit in one byte. For example, the byte one hundred seventy-eight is written two, zero, one, two, one in base three; subtracting one from each digit gives the five weights one, minus one, zero, one, zero. This format uses one point six bits per weight on average, twenty times less than thirty-two-bit floats.

Weights can technically be quantized to a single bit each, eight per byte, with each bit representing minus one or plus one, or sometimes zero or one. Reasonable accuracy is very hard to reach with such severe quantization.

---

Reduced precision models.

Converting all of a model's parameters to sixteen-bit floats is called half precision, and a PyTorch model's half method does it. It halves the size of a trained model, usually without much impact on quality. Many GPUs have sixteen-bit float optimizations, and less data moves between the CPU and the GPU, so the model typically runs almost twice as fast.

A half-precision model takes sixteen-bit inputs and produces sixteen-bit outputs. A sixteen-bit model can also be built and trained from the start, by creating its tensors and modules with the float16 data type.

Training in sixteen bits can cause problems, because float16 has a limited dynamic range, meaning the ratio between the largest and smallest positive representable values. The smallest positive value is about six times ten to the minus eight, and the largest is sixty-five thousand five hundred four. Any gradient update smaller than about six times ten to the minus eight underflows: it is rounded down to zero and ignored. Any value larger than about sixty-five thousand overflows to infinity, and once some weights are infinite, the loss becomes infinite or NaN and training fails.

To avoid underflows, the loss can be scaled up by a large factor, for example multiplied by two hundred fifty-six. The backward pass then scales up the gradients by the same factor, keeping them above the smallest representable value. The gradients must be scaled back down before the optimizer step, and at that point they may underflow again. And scaling the loss up too much causes overflows.

If no scaling factor avoids both problems, bfloat16 can be used instead of float16. It has more exponent bits: its smallest value is about nine point two times ten to the minus forty-one and its largest about three point four times ten to the thirty-eight, so significant gradient updates are less likely to be ignored and reasonable values are less likely to become infinite.

Bfloat16 has two drawbacks. Historically it has had less hardware support, although this is improving. And it has fewer fraction bits, so a gradient update can be lost when the parameter is much larger, which can make training stall. For example, adding an update of zero point zero four five to a parameter of one hundred twenty-three should give one hundred twenty-three point zero four five, but bfloat16 lacks the fraction bits for those digits and rounds the result back to one hundred twenty-three, ignoring the update completely. Regular float16 gives one hundred twenty-three point zero six two five, not exact, but at least a step in the right direction.

If both float16 and bfloat16 still cause convergence problems, mixed-precision training is the next option.

---

Mixed-precision training.

Mixed-precision training was proposed by Baidu and Nvidia researchers in 2017 to fix the issues often seen with sixteen-bit training. It keeps a primary copy of the model parameters as thirty-two-bit floats. At each training iteration, it creates a sixteen-bit copy of the parameters and uses it for the forward pass. The loss is scaled up by a large factor to avoid underflows, and the backward pass produces scaled gradients. Then MPT switches back to thirty-two-bit precision to scale the gradients back down, where the greater precision avoids underflow. Finally, it uses the gradients to perform an optimizer step on the thirty-two-bit primary parameters. Doing the update in thirty-two bits, with twenty-three fraction bits, keeps small updates from being lost on much larger parameters.

MPT offers almost all the benefits of sixteen-bit training without the instabilities. The parameters take fifty percent more space than in thirty-two-bit training, because of the extra sixteen-bit copy. But during training most of the RAM holds activations, not parameters, so in practice MPT needs just a bit more than half the RAM of regular thirty-two-bit training, and it typically runs twice as fast. After training, the thirty-two-bit parameters are no longer needed; converting them to sixteen bits gives a pure sixteen-bit model. It does not always speed up training, depending on the model, batch size, and hardware, but most large transformers are trained with MPT.

The loss scaling factor must be large enough to avoid underflows and small enough to avoid overflows. One way to find it is to train in thirty-two-bit precision for a little while, if there is enough RAM, and measure the gradient statistics. Another is to adapt the factor dynamically during training. If some gradients are infinite or NaN, an overflow occurred, so the factor is reduced, for example halved, and that training step is skipped. If no overflow is detected, the factor is gradually increased, for example doubled every two thousand training steps. PyTorch's gradient scaler class implements this approach.

PyTorch's autocast function returns a context in which many operations automatically run in sixteen-bit precision. These include the operations that benefit most, such as matrix multiplications and convolutions, but not reductions such as sums, because running them in half precision brings no significant benefit and can damage precision.

A mixed-precision training loop creates a gradient scaler with a large initial scale, two to the power sixteen. For each batch, it runs the forward pass and computes the loss inside an autocast context, using float16 or bfloat16. Then the scaler scales the loss before the backward pass, performs the optimizer step, and updates its scaling factor, and the gradients are reset. The Hugging Face Transformers trainer activates mixed precision with its fp16 or bf16 option.

---

Quantization.

Quantization means mapping continuous values to discrete ones. In deep learning, it usually means converting parameters, and often activations, from floats to integers, typically from thirty-two-bit floats to eight-bit integers.

Some embedded devices, such as the ARM Cortex-M0, do not support floating-point operations at all, partly to reduce cost and energy use, so models must be fully quantized, weights and activations, to run on them. Modern smartphones do support floating point but still benefit a lot: eight-bit integer operations are two to four times faster and use five to ten times less energy than thirty-two-bit float operations.

The simplest approach is linear quantization. Nonlinear methods come later in the appendix.

---

Asymmetric linear quantization.

Linear quantization dates back to digital signal processing in the 1950s, and has become important in machine learning as models have grown gigantic. It has two variants, asymmetric and symmetric.

In asymmetric linear quantization, floats are mapped linearly to unsigned bytes from zero to two hundred fifty-five, or more generally from zero to two to the n minus one for n-bit integers. If the weights range from minus zero point one to zero point six, then minus zero point one maps to zero, zero maps to thirty-six, zero point one maps to seventy-two, and zero point six maps to two hundred fifty-five.

The mapping uses two quantization parameters. The scale is the range of the values, the maximum minus the minimum, divided by two to the n minus one. The zero point, also called the quantization bias, is minus the minimum divided by the scale, rounded. Each quantized integer is the original float divided by the scale, rounded, plus the zero point, and then clamped between zero and two to the n minus one.

The range of the weights is known, because weights do not change after training. The range of activations depends on the inputs. For each activation to quantize, such as each layer's inputs, there are two options. Dynamic quantization computes the minimum and maximum on the fly for each new input batch. Static quantization runs a calibration dataset through the model once to find the typical range of the activations, then uses that range for all later batches. Static quantization is faster but less precise.

Dequantization approximately recovers the original float: it is the scale times the quantized integer minus the zero point. For example, seventy-two dequantizes to about zero point zero nine eight eight, close to zero point one. The difference between the dequantized value and the original is the quantization noise. With eight-bit quantization, the noise usually degrades accuracy slightly. With six bits, four bits, or fewer, it can hurt more, especially because the effect accumulates: the deeper the network, the stronger the impact.

The formula guarantees that a float equal to zero is quantized to the zero point and dequantized back to exactly zero. This matters for sparse models, where many weights are zero, and for activation functions such as ReLU, which produce many zeros.

In PyTorch, the quantize per tensor function creates a quantized tensor, which holds the quantized integers along with the scale and the zero point, and its dequantize method converts it back to floats.

Quantizing a model to eight bits divides its size by almost four. A convolutional layer with sixty-four kernels of three by three and thirty-two input channels has eighteen thousand four hundred thirty-two parameters, ignoring biases. That is seventy-three thousand seven hundred twenty-eight bytes as thirty-two-bit floats, and eighteen thousand four hundred thirty-two bytes after quantization, plus eight bytes for the scale and the zero point, each stored as a thirty-two-bit float.

PyTorch also has a quantize per channel function, which quantizes each channel separately. It is more precise but needs a little more space for the extra quantization parameters.

---

Symmetric linear quantization.

When the floats are roughly symmetric around zero, symmetric linear quantization maps them to integers between minus one hundred twenty-seven and plus one hundred twenty-seven, or more generally between minus r and plus r, where r is two to the power n minus one, minus one. Each quantized integer is the float divided by the scale, rounded, and the scale is the largest absolute value in the tensor divided by r. There is no zero point, and zero always maps to zero.

In PyTorch, symmetric quantization uses the same quantize per tensor function, with a zero point of zero and a signed eight-bit quantized data type.

PyTorch uses restricted symmetric quantization, which excludes minus one hundred twenty-eight to keep the range symmetric. Unrestricted symmetric quantization, used by some other implementations, also allows minus one hundred twenty-eight.

Symmetric mode is often a bit faster than asymmetric mode, because there is no zero point to handle. But if the values are not symmetric, part of the integer range is wasted: if all the weights are positive, only zero to one hundred twenty-seven is used. So symmetric mode can be a bit less precise. In practice, symmetric mode is generally preferred for weights, which are often fairly symmetric, and asymmetric mode for activations, especially after ReLU, which outputs only nonnegative values.

A model can be quantized in two ways. Post-training quantization, or PTQ, quantizes a trained model. Quantization-aware training, or QAT, trains or fine-tunes the model with fake quantization so it gets used to the noise.

---

Post-training dynamic quantization.

PyTorch's AO package, for architecture optimization, includes tools for pruning, sparsity, and quantization, with two ways to quantize a trained model: dynamic and static.

Dynamic quantization is best for multilayer perceptrons, recurrent neural networks, and transformers. It first requires choosing a quantization engine. PyTorch supports the Facebook General Matrix Multiplication engine, or FBGEMM, for x86 CPUs; a newer x86 engine that supports recent x86 CPUs but is less battle-tested; and the Quantized Neural Networks Package, or QNNPACK, for ARM and mobile devices. PyTorch offers no engine for CUDA or other hardware accelerators, but other libraries do, such as bitsandbytes.

Once the engine is set, the quantize dynamic function takes the trained model, the types of layers to quantize, typically just the linear and recurrent layers, and the quantized data type, and returns a ready-to-use quantized model. Each linear layer is replaced with a dynamic quantized linear layer that has eight-bit integer weights. It behaves like a regular linear layer, with float inputs and outputs. Internally, it quantizes its inputs on the fly, recomputing the scale and zero point for each batch, performs the matrix multiplication with integers only, using thirty-two-bit integer accumulators, and dequantizes the result so the next layer gets floats.

---

Post-training static quantization.

Static quantization is best for convolutional neural networks and for maximum inference speed. It is required on edge devices without a floating-point unit, since they do not support floats at all. Weights and activations are both prepared ahead of time, for all layers: weights are quantized once, and activations get a calibration step to find their typical range.

The thirty-two-bit model is built with a quant stub layer at the start and a dequant stub layer at the end, which initially just pass data through, and is trained normally.

Next, the model's quantization configuration is set to the default configuration for the chosen engine. It specifies the quantized data type, such as signed eight-bit integers; the quantization scheme, such as symmetric linear quantization per tensor; and two observer functions that watch the weights and the activations to determine their ranges.

Then PyTorch's prepare function uses the weight observer to compute the scales and zero points for the weights. The inputs are not known yet, so it inserts activation observers into the model, after the quant stub and the linear layers. For calibration, a representative sample of input batches, like those the model will see in production, is passed through the model so the observers can track the activation ranges.

Finally, PyTorch's convert function removes the observers and replaces the layers with quantized versions. The quant stub becomes a quantize layer that quantizes the inputs, the linear layers become quantized linear layers, and the dequant stub becomes a dequantize layer that dequantizes the outputs.

Observers can track the minimum and maximum per tensor or per channel, keep an exponential moving average of them to reduce the impact of outliers, or record a histogram to find the range that minimizes quantization error. The defaults are usually fine.

The result is used like a normal model, with float inputs and outputs, but it works entirely with integers internally, which makes it light and fast. Deployment options for mobile or embedded devices include ExecuTorch, PyTorch's lightweight edge runtime; ONNX Runtime; TFLite or TFLite Micro; and TVM or microTVM.

The PyTorch team has also released a separate library, PyTorch-native Architecture Optimization, or TorchAO, designed as a robust and extensible model optimization framework. Many features of PyTorch's AO package are expected to move to it. TorchAO already supports four-bit weights and per-block quantization, in which each tensor is split into small blocks that are quantized independently, trading space for precision.

Post-training quantization, dynamic or static, shrinks and speeds up models significantly, but it degrades their accuracy. The loss is worse when quantizing to four bits or fewer, and worse for static quantization than for dynamic quantization, which can at least adapt to each input batch. When the accuracy drop is unacceptable, quantization-aware training is the next option.

---

Quantization-aware training.

Quantization-aware training was introduced in a 2017 paper by Google researchers. The idea is to add fake quantization noise during training so that the model learns to cope with it. After training, the model is quantized for real, and it remains fairly accurate. QAT also allows more aggressive quantization, down to four bits or fewer, without losing too much accuracy.

Fake quantization noise is added to a weight by quantizing it and immediately dequantizing it. A weight of zero point four two might be quantized to the four-bit integer seven and dequantized back to zero point three nine. That is exactly the noise the model would get if it were really quantized. This fake quantization runs at every training step, and it can also be applied to some activations, such as each layer's output.

Rounding to the nearest integer has a gradient of zero, or an undefined gradient at integer boundaries, so gradient descent cannot make progress through it. The straight-through estimator, or STE, sidesteps this. During the backward pass, the fake quantization operation is treated as the identity function, so gradients flow straight through it unchanged. This works because the loss landscape is generally fairly smooth locally, so gradients within a small region around the quantized value, including at the original value, are likely to be similar.

In PyTorch, QAT starts by setting the model's quantization configuration to the default QAT configuration for the chosen engine. The prepare QAT function then adds fake quantization operations to the model, along with observers that determine the usual range of the activations. The model is trained normally, switched to evaluation mode, and passed to the convert function to be truly quantized.

QAT can also be used just to fine-tune a pretrained model for a few epochs, typically with a lower learning rate to avoid damaging the pretrained weights.

PyTorch's AO package is designed mainly for CPUs. To run a large language model, or LLM, on a GPU that does not quite have enough RAM, the options include TorchAO, which has growing GPU support, and the bitsandbytes library.

---

The bitsandbytes library.

The bitsandbytes library, created by Tim Dettmers, makes it easier to train and run large models on GPUs with limited video memory, or VRAM. It provides quantization tools, including four-bit quantization and block-wise quantization; memory-efficient versions of popular optimizers, such as Adam and AdamW, that operate on eight-bit tensors; and custom CUDA kernels written for eight-bit and four-bit quantized models, for maximum speed. It is designed for Nvidia GPUs, with some limited support for CPUs and AMD GPUs.

The book's example applies post-training static quantization down to four bits through the Hugging Face Transformers library. It creates a bitsandbytes configuration that requests four-bit loading, the NF4 quantization type, and bfloat16 as the compute data type. It passes this configuration when downloading a pretrained model: TinyLlama, a one point one billion parameter version of Llama fine-tuned for chat.

Under the hood, Transformers uses bitsandbytes to quantize the weights to four bits as they are loaded into the GPU, with no extra step. During inference, weights are dequantized on the fly to the compute data type, bfloat16 here, used at that higher precision, and dropped as soon as they are no longer needed, so memory use stays low.

---

NF4 and QLoRA.

Four-bit Normal Float, or NF4, is a nonlinear four-bit scheme. Each of its sixteen possible integer values represents a specific float between minus one and plus one. These floats are not equally spaced, as in linear quantization; they correspond to quantiles of a normal distribution centered on zero, so they are closer together near zero. Model weights tend to follow a normal distribution centered on zero, so the extra precision near zero improves accuracy.

NF4 was introduced as part of QLoRA, which quantizes a frozen pretrained model with NF4, then fine-tunes it with low-rank adaptation adapters, or LoRA adapters, along with activation checkpointing. This drastically reduces VRAM use and compute. The authors fine-tuned a sixty-five billion parameter model on a single GPU with forty-eight gigabytes of RAM, with only a small accuracy drop.

Activation checkpointing reduces VRAM use overall, but it can cause memory spikes on batches with long sequences. To handle them, the QLoRA authors introduced paged optimizers, which rely on Nvidia unified memory: the CUDA driver automatically moves pages of data from GPU VRAM to CPU RAM when needed. They also used double quantization, meaning the quantization parameters themselves are quantized, to save a bit more VRAM.

---

Pre-quantized models.

Many popular pretrained models have already been quantized and published, especially on the Hugging Face Hub. For example, Tom Jobbins, known on Hugging Face as The Bloke, has published thousands of quantized models. Many were quantized with one of three modern methods.

Generative pre-training quantization, or GPTQ, is a post-training method, usually down to four bits, that treats quantization as an optimization problem. It goes through the layers one by one. For each layer, it optimizes the four-bit weights to minimize the mean squared error between the layer's original outputs, computed with full-precision weights, and its approximate outputs, computed with four-bit weights. The approximate outputs are then passed to the next layer, and the process repeats up to the output layer. GPTQ quantizes only the weights, not the activations, which is called weight-only quantization; it is great for inference, but not for training. The Hugging Face Optimum library or the GPTQModel library can apply it.

Activation-aware weight quantization, or AWQ, improves the accuracy of block-wise weight-only quantization, typically to four bits, by preserving the precision of the most important weights. To find these salient weights, it runs a calibration dataset through the model and finds the largest activations in each quantization group, for example the top zero point one to one percent; the corresponding weights are salient. The authors found that keeping the salient weights in float16 greatly reduces the model's perplexity, a common metric equal to the exponential of the cross-entropy. But mixing four-bit and sixteen-bit weights is not hardware-friendly. So AWQ instead scales the salient weights up by some factor and adds an operation that scales the corresponding activations down, which can usually be fused into the previous operation. AWQ searches for the scaling factor that gives the lowest quantization error, and the Hugging Face Optimum library implements it.

GGUF, the GPT-Generated Unified Format, is a binary file format designed to store LLMs efficiently. It was introduced by Georgi Gerganov, the creator of llama dot cpp, and it supersedes earlier formats such as GGML, GGMF, and GGJT. A GGUF file includes the weights, the tokenizer, special tokens, the model architecture, the vocabulary size, and other metadata. Llama dot cpp provides quantizers, such as its quantize tool, that convert model weights to one of GGUF's quantized formats, such as Q4 K M. Q4 means four-bit quantization. K means per-block quantization, typically with thirty-two or sixty-four weights per block depending on the format. M means medium size and precision for that level, with S for small and L for large as the other options. Newer options include importance-aware quantization, or IQ, which uses techniques such as nonlinear quantization, and ternary quantization, or TQ.

Hugging Face Hub repositories are backed by Git, and quantized models are often stored in a branch other than main, so loading one may require selecting the branch, tag, or commit, and for GGUF models, the file name.

---

Other ways to shrink models.

Beyond reduced precision, mixed precision, and quantization, there are other ways to fit large models on limited hardware.

The architecture can be adjusted before training, by reducing the number of layers or of neurons per layer, or by sharing weights across layers, as in the ALBERT model.

Model pruning shrinks a large trained model by removing some of its weights, for example those with the smallest magnitude or the smallest effect on the loss. Whole channels, layers, or attention heads can also be removed. PyTorch's pruning utilities and the Hugging Face Optimum library implement it.

Distillation uses a large trained model as a teacher to train a smaller model.

Layer fusion merges layers to remove redundancy. A batch-norm layer performs a linear operation, so when it immediately follows a linear layer, the two can be fused into a single linear layer. Similarly, a convolutional layer followed by a batch-norm layer can be fused into a single convolutional layer. This only works after training, because the batch-norm layer must compute running averages during training. PyTorch and the Hugging Face Optimum library both provide layer fusion. Layers should be fused before quantizing, since fewer layers mean less quantization noise.

Low-rank approximation replaces a large matrix with the product of two smaller ones. For example, a linear layer with ten thousand inputs and twenty thousand outputs can be replaced by a layer from ten thousand to one hundred, followed by a layer from one hundred to twenty thousand. This cuts the parameters from about two hundred million to three million, and drastically reduces computation. The intermediate dimension is a hyperparameter that balances accuracy against size. The approximation can be made after training, by factorizing the weight matrix with singular value decomposition, or SVD.

---

Summary.

Floating-point formats differ in how many bits go to the exponent, which sets the range, and to the fraction, which sets the precision. Float16 has a limited range, so small gradients underflow and large values overflow. Bfloat16 has a wider range but less precision, so small updates to large parameters can be lost.

Converting a trained model to sixteen bits halves its size and nearly doubles its speed. Mixed-precision training keeps a thirty-two-bit primary copy of the weights, runs the forward and backward passes in sixteen bits with a scaled loss, and applies the updates in thirty-two bits.

Quantization maps floats to small integers. Asymmetric linear quantization uses a scale and a zero point and suits activations; symmetric linear quantization uses only a scale and suits weights. Activations are quantized dynamically, per batch, or statically, using a calibration dataset.

Post-training quantization shrinks a trained model but costs some accuracy. Quantization-aware training adds fake quantization noise during training, with the straight-through estimator for gradients, so the model tolerates more aggressive quantization.

For large language models on GPUs, bitsandbytes provides four-bit NF4 quantization, the basis of QLoRA. Pre-quantized models are widely available, made with methods such as GPTQ, AWQ, and GGUF. Pruning, distillation, layer fusion, and low-rank approximation shrink models further.
