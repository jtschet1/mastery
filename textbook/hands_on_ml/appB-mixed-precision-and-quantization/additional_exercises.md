# Appendix B — Mixed Precision and Quantization

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The book has no exercises for this appendix, so every
card for it is here.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Estimate the memory needed to train a 1-billion-parameter model in fp32 with Adam. Why does this motivate lower precision and quantization?

Each fp32 parameter takes 4 bytes, so the weights alone need 4 GB; inference also needs room for the current batch's activations. Training adds a gradient per parameter (another 4 GB) and Adam's two extra values per parameter (8 GB): about 16 GB before counting the activations saved for the backward pass, which often dominate. Big tensors also cost time moving data between CPU and GPU, plus storage, download time and energy.

Switching to 16-bit floats halves the size of the weights and activations. Quantizing the weights to 8-bit integers divides their size by almost 4 (almost 8 with 4 bits), and integer arithmetic is also faster and more energy efficient: on smartphones, int8 operations are typically 2 to 4 times faster than fp32 and use 5 to 10 times less energy.

## 2. How is a float32 number laid out in bits, and how do you compute its value?

IEEE 754 float32 has 1 sign bit S, 8 exponent bits E (0 to 255) and 23 fraction bits F:
- Normalized numbers (E from 1 to 254, the usual case): v = (−1)ˢ · 2ᴱ⁻¹²⁷ · (1 + F · 2⁻²³). The exponent is stored with a bias of 127, and the last factor, the significand, holds the significant digits.
- Subnormal numbers (E = 0, F > 0): v = (−1)ˢ · F · 2⁻¹⁴⁹, covering the tiniest magnitudes, down to about 1.4 × 10⁻⁴⁵.
- E = 0 and F = 0 gives ±0; E = 255 gives ±∞ if F = 0, and NaN otherwise.

The largest float32 is about 3.4 × 10³⁸. Other float formats only change the split: float16 has 5 exponent bits (bias 15) and 10 fraction bits, bfloat16 has 8 and 7, and fp8 E4M3 has 4 and 3. Exponent bits buy range; fraction bits buy precision.

## 3. How are weights quantized to fewer than 8 bits usually stored in memory?

They're packed several per byte. For example, two 4-bit values fit in one unsigned byte using bit shifts and masks:

```python
packed = torch.tensor((12 << 4) | 5, dtype=torch.uint8)  # stores 12 and 5
high, low = packed >> 4, packed & 0xF  # unpacks 12 and 5
```

Likewise, four 2-bit values fit in a byte. Ternary weights (−1, 0 or +1) can be stored five per byte by adding 1 to each and treating them as base-3 digits: since 3⁵ = 243 ≤ 256, that's 1.6 bits per weight, 20 times less than fp32. You can even go down to 1 bit per weight (8 per byte, each bit meaning −1 or +1), but keeping reasonable accuracy with such severe quantization is very hard.

## 4. How do you convert a PyTorch model to half precision, or build it in 16 bits from the start, and what do you need to watch out for?

- model.half() converts all the parameters of a trained fp32 model to float16. This halves the model's size, usually with little impact on quality, and it often runs almost twice as fast on GPUs with 16-bit support.
- The model then expects 16-bit inputs and produces 16-bit outputs, e.g., X = torch.rand(3, 10, dtype=torch.float16); feeding it fp32 inputs raises a dtype mismatch error.
- To build a 16-bit model directly, pass dtype=torch.float16 when creating each tensor or layer, such as nn.Linear(10, 100, dtype=torch.float16), or call torch.set_default_dtype(torch.float16), which affects every tensor and module created afterward.
- With the Transformers library, from_pretrained(..., dtype="auto") picks the best float type for the checkpoint and your hardware.

Shrinking a trained model this way is fairly safe, but training directly in float16 can fail to converge because of underflows and overflows.

## 5. Compare float16 and bfloat16 for training. What can go wrong with each?

Both use 16 bits, split differently:
- float16 (5 exponent bits, 10 fraction bits) is more precise but only covers about 6 × 10⁻⁸ to 65,504. Gradient updates smaller than about 6 × 10⁻⁸ underflow to zero and are ignored, while values above 65,504 overflow to infinity, which soon makes the loss infinite or NaN.
- bfloat16 (8 exponent bits, 7 fraction bits) has roughly float32's range, about 9.2 × 10⁻⁴¹ to 3.4 × 10³⁸, so underflow and overflow are rarely an issue. But its low precision can swallow small updates to large weights: 123 + 0.045 rounds back to 123 in bfloat16 (float16 gives 123.0625), so training can stall. It has also historically had less hardware support.

If you still hit convergence issues with both, switch to mixed-precision training.

## 6. What is loss scaling, why does float16 training need it, and how does dynamic loss scaling choose the factor?

Many gradients are smaller than float16's smallest positive value (about 6 × 10⁻⁸), so they underflow to zero. Multiplying the loss by a large factor (e.g., 2¹⁶) multiplies every gradient by the same factor during backpropagation, lifting them into range. The gradients must be scaled back down before the optimizer step, ideally in fp32 so they don't underflow again. Too large a factor causes overflows instead: infinite or NaN gradients.

You can choose a fixed factor from gradient statistics measured during a short fp32 run, or use dynamic loss scaling: if any gradient is infinite or NaN, skip that optimizer step and reduce the factor (e.g., halve it); otherwise, increase it periodically (e.g., double it every 2,000 steps). PyTorch's torch.amp.GradScaler implements this.

## 7. Walk through one iteration of mixed-precision training. Why does it use less memory than fp32 training despite keeping two copies of the weights?

1. Keep a primary copy of the parameters in fp32, and make a 16-bit copy of them.
2. Run the forward pass with the 16-bit weights, so the activations are 16-bit too.
3. Scale up the loss to prevent gradient underflow, and backpropagate in 16 bits.
4. Switch to fp32 to scale the gradients back down.
5. Apply the optimizer step to the fp32 primary weights, so small updates to large weights aren't rounded away (fp32 has 23 fraction bits).

The parameters take 50% more memory than in fp32 training, but most training memory goes to activations, which are now 16-bit, so mixed precision needs only a bit more than half the memory, and it typically runs about twice as fast (depending on the model, batch size and hardware). After training, you can drop the fp32 copy and keep a pure 16-bit model.

## 8. Write a mixed-precision training loop in PyTorch with torch.autocast and GradScaler, and explain what each call does.

Only the forward pass and the loss computation go inside the autocast context:

```python
from torch.amp import GradScaler

scaler = GradScaler(device="cuda", init_scale=2.0**16)
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(X_batch), y_batch)
    scaler.scale(loss).backward()  # backprop the scaled loss
    scaler.step(optimizer)  # unscale the grads; skip the step if inf/NaN
    scaler.update()  # adjust the scale factor
    optimizer.zero_grad()
```

Under autocast the parameters stay fp32, but the operations that benefit most from 16 bits, such as matrix multiplications and convolutions, run in float16, while operations like reductions (e.g., torch.sum()), which gain little and could lose precision, don't. With the Hugging Face Trainer, just set fp16=True or bf16=True in TrainingArguments.

## 9. Explain asymmetric linear quantization to n-bit unsigned integers: the scale, the zero point, and dequantization.

With a = minᵢ wᵢ and b = maxᵢ wᵢ, the range [a, b] maps linearly onto the integers 0 to 2ⁿ − 1:

s = (b − a) / (2ⁿ − 1), z = −round(a / s), qᵢ = round(wᵢ / s) + z (clamped to [0, 2ⁿ − 1])

- s, the scale, is the float step between consecutive integers.
- z, the zero point, is the integer that represents 0.0.
- Dequantization approximately recovers each float: wᵢ ≈ s · (qᵢ − z). The error is the quantization noise; it grows as n shrinks and accumulates through deep networks.

For example, with 8 bits and weights in [−0.1, 0.6], s ≈ 0.002745 and z = 36, so 0.1 becomes 72, which dequantizes to about 0.0988. A useful property: 0.0 maps exactly to z and comes back as exactly 0.0, which matters for sparse weights and for the many zeros that ReLU outputs.

## 10. How does symmetric linear quantization differ from asymmetric quantization, and when would you use each?

Symmetric quantization maps values to signed integers centered on zero, with no zero point:

qᵢ = round(wᵢ / s), with s = maxᵢ |wᵢ| / (2ⁿ⁻¹ − 1)

With 8 bits that's −127 to +127 (leaving out −128 keeps the range symmetric), and 0.0 always maps to 0. The trade-offs:
- Symmetric is simpler and often a bit faster, since there's no zero point to handle. But if the values aren't centered on zero, part of the integer range goes unused: all-positive values would only use 0 to 127, so precision suffers.
- Asymmetric fits the actual [min, max] range, so it's more precise for skewed data.

In practice, symmetric quantization is generally preferred for weights, which tend to be fairly symmetric around zero, and asymmetric quantization for activations, especially after ReLU, which outputs only nonnegative values.

## 11. How do you quantize and dequantize a tensor in PyTorch with torch.quantize_per_tensor()?

Pass the float tensor, the scale, the zero point and a quantized dtype. You get a quantized tensor that stores the integers together with the scale and zero point:

```python
w = torch.tensor([0.1, -0.1, 0.6, 0.0])
s = (w.max() - w.min()) / 255.  # asymmetric, 8 bits
z = -(w.min() / s).round()
qw = torch.quantize_per_tensor(w, scale=s, zero_point=z, dtype=torch.quint8)
qw.dequantize()  # tensor([ 0.0988, -0.0988,  0.6012,  0.0000])

s_sym = w.abs().max() / 127.  # symmetric: zero point 0, signed integers
qw_sym = torch.quantize_per_tensor(w, scale=s_sym, zero_point=0,
                                   dtype=torch.qint8)
```

torch.quint8 holds unsigned 8-bit integers (for asymmetric quantization) and torch.qint8 signed ones (for symmetric quantization). Printing a quantized tensor shows its dequantized values along with the scheme (per_tensor_affine), scale and zero_point, while qw.int_repr() returns the raw integers (72, 0, 255 and 36 here). torch.quantize_per_channel() instead gives each channel its own scale and zero point: more precise, at the cost of storing more quantization parameters.

## 12. What's the difference between dynamic and static quantization, and when is each a good fit?

Weights are fixed after training, so they can always be quantized ahead of time. The difference is how activations are handled, since their range depends on the inputs:
- Dynamic quantization computes each activation's range, and hence its scale and zero point, on the fly for every batch. It needs no calibration data and adapts to each input, so it loses less accuracy, but the extra work makes it slower. It's best for MLPs, RNNs and transformers.
- Static quantization runs a representative calibration dataset through the model once to estimate typical activation ranges, then reuses those fixed parameters for all inputs, so the whole network can compute with integers. It's faster but less precise, which makes it best for CNNs and maximum inference speed, and it's mandatory on edge devices without a floating-point unit.

Both degrade accuracy somewhat, especially at 4 bits or fewer.

## 13. How do you apply post-training dynamic quantization with torch.ao.quantization, and what does the quantized model do at inference?

Pick the quantization engine for the CPU you'll run on ("x86" or "fbgemm" for x86 CPUs, "qnnpack" for ARM and mobile), then tell quantize_dynamic() which layer types to quantize:

```python
from torch.ao.quantization import quantize_dynamic

torch.backends.quantized.engine = "x86"  # or "qnnpack" on ARM
qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
y_pred = qmodel(torch.randn(3, 10))  # float inputs and outputs
```

Each nn.Linear becomes a DynamicQuantizedLinear layer with int8 weights. At inference, it quantizes its float inputs on the fly (computing a new scale and zero point for each batch), multiplies integers using 32-bit integer accumulators, and dequantizes the result, so the next layer receives floats as usual. RNN layer types can be included in the set too.

Gotcha: torch.ao.quantization's engines only target CPUs (there's none for CUDA or other accelerators), so for GPUs use a library such as bitsandbytes or the separate TorchAO library.

## 14. Walk through post-training static quantization with torch.ao.quantization.

The whole workflow takes a few lines:

```python
from torch.ao.quantization import get_default_qconfig, QuantStub, DeQuantStub

model = nn.Sequential(QuantStub(), nn.Linear(10, 100), nn.ReLU(),
                      nn.Linear(100, 1), DeQuantStub())
# [...] train the model normally, in fp32
model.qconfig = get_default_qconfig("x86")
torch.ao.quantization.prepare(model, inplace=True)
for X_batch, _ in calibration_loader:
    model(X_batch)
torch.ao.quantization.convert(model, inplace=True)
```

1. QuantStub and DeQuantStub mark where tensors enter and leave the quantized part of the model; until conversion they just pass data through.
2. The qconfig for the chosen engine specifies the quantized dtype, the quantization scheme, and the observers that track the ranges of weights and activations.
3. prepare() inserts the observers into the model, e.g., MinMaxObserver, or HistogramObserver, which finds the range that minimizes quantization error.
4. Calibration: running representative batches lets the observers record typical activation ranges.
5. convert() removes the observers and swaps in quantized modules (Quantize, QuantizedLinear, DeQuantize): the model still takes and returns floats but computes with integers internally.

## 15. How does quantization-aware training work, how do gradients get through the rounding, and how do you set it up with torch.ao.quantization?

QAT trains the model, or fine-tunes a pretrained one for a few epochs at a low learning rate, with fake quantization: in the forward pass, weights and some activations are quantized and immediately dequantized, so the model experiences exactly the noise real quantization will add and learns to cope with it. This loses less accuracy than post-training quantization and makes aggressive quantization (4 bits or less) practical.

Rounding has zero gradient almost everywhere, so backpropagation uses the straight-through estimator: the backward pass treats fake quantization as the identity function, letting gradients through unchanged. This works because the loss surface is fairly smooth locally, so the gradient at the rounded value is close to the gradient at the original one.

In PyTorch: set model.qconfig = get_default_qat_qconfig(engine), call torch.ao.quantization.prepare_qat(model, inplace=True) to insert fake quantization and observers, train normally, then call convert(model.eval(), inplace=True).

## 16. How do you load an LLM in 4 bits on a GPU with bitsandbytes and the Transformers library, and what happens at inference?

Describe the quantization with a BitsAndBytesConfig and pass it to from_pretrained():

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto",
    quantization_config=bnb_config)
```

The weights are quantized to 4 bits as they're loaded onto the GPU, with no extra step, and you then use the model normally (e.g., call generate()). Whenever some weights are needed during inference, they're dequantized on the fly to the compute dtype (bfloat16 here), the computation runs in that precision, and the dequantized copy is dropped, so memory use stays low. Setting bnb_4bit_use_double_quant=True also quantizes the quantization constants, saving a bit more memory. bitsandbytes is designed for NVIDIA GPUs (with limited CPU and AMD support), and it also offers 8-bit versions of optimizers such as Adam.

## 17. What is NF4 quantization, and which techniques does QLoRA combine to fine-tune huge models on a single GPU?

NF4 (4-bit NormalFloat) is a nonlinear 4-bit scheme: instead of being evenly spaced, its 16 values between −1 and +1 sit at the quantiles of a zero-centered normal distribution, so they're denser near zero. Trained weights roughly follow such a distribution, so NF4 spends its precision where most weights are, which reduces the quantization error.

QLoRA combines:
- a frozen pretrained model quantized to NF4;
- small trainable LoRA adapters, the only weights updated during fine-tuning;
- activation checkpointing, to save activation memory;
- paged optimizers, which use NVIDIA unified memory to move pages of data automatically between GPU and CPU RAM, absorbing memory spikes on long sequences;
- double quantization, which quantizes the quantization parameters themselves.

With it, the authors fine-tuned a 65-billion-parameter model on a single 48 GB GPU with only a small accuracy drop.

## 18. How do GPTQ and AWQ quantize LLM weights to 4 bits while limiting the accuracy loss?

Both are post-training, weight-only methods: weights are stored in 4 bits and dequantized when needed, while activations stay in floating point, which suits inference but not training.
- GPTQ treats quantization as an optimization problem, one layer at a time: it finds the 4-bit weights that minimize the MSE between the layer's outputs with the original weights and with the quantized ones, then feeds the approximate outputs to the next layer and repeats.
- AWQ (activation-aware weight quantization) protects the salient weights: those that multiply the largest activations on a calibration dataset (roughly the top 0.1% to 1%). Keeping them in float16 would work but isn't hardware-friendly, so AWQ scales them up before quantizing and scales the matching activations down (usually folded into the previous operation), searching for the scale factor with the lowest quantization error.

You can apply both with the Hugging Face Optimum library.

## 19. What is GGUF, what does a quantization type like Q4_K_M mean, and how do you load pre-quantized models from the Hugging Face Hub?

GGUF is the binary file format used by llama.cpp and the tools built on it, such as Ollama and LM Studio. A single file bundles the weights with the tokenizer, special tokens, model architecture and other metadata. Its quantization types have names like Q4_K_M: Q4 means 4-bit, K means per-block quantization (each small block of weights gets its own quantization parameters), and M means the medium size and precision option at that bit width (the others being S and L). Newer types include IQ (importance-aware) and TQ (ternary).

Every Hub repository is a Git repo, and quantized variants often live on a separate branch, so pass revision= (a branch, tag or commit hash) to from_pretrained(), after checking the model card for what's available. For a GGUF model, pass the file name with gguf_file=, e.g., from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file="tinyllama-1.1b-chat-v1.0.Q6_K.gguf").

## 20. Besides lower precision and quantization, what other techniques can shrink a model?

- Architecture changes before training: fewer layers or fewer neurons per layer, or weights shared across layers (as in ALBERT).
- Pruning: remove the weights with the smallest magnitude or the smallest effect on the loss, or entire channels, layers or attention heads (torch.nn.utils.prune or Hugging Face Optimum).
- Distillation: train a small student model to reproduce a large teacher model's outputs.
- Layer fusion: after training, fold a batch-norm layer into the linear or convolutional layer just before it, since at inference both are linear operations (torch.quantization.fuse_modules(), or Optimum). Fuse before quantizing: fewer layers means less quantization noise.
- Low-rank factorization: replace a big weight matrix with the product of two thin ones, e.g., Linear(10_000, 20_000) with Linear(10_000, 100) followed by Linear(100, 20_000), cutting about 200 million parameters to 3 million, and the compute with them. The middle size trades accuracy for size, and a trained layer can be factorized with SVD.
