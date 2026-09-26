# Chapter 10 — Building Neural Networks with PyTorch

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/10_neural_nets_with_pytorch.ipynb`, converted to plain
text for the flashcard app.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 13. Use autograd to find the gradient of f(x, y) = sin(x²y) at (x, y) =
  (1.2, 3.4).
- 14. Write a custom Dense module (Linear + ReLU), first with nn.Linear and
  nn.ReLU, then with nn.Parameter and relu().
- 15. Build and train an MLP classifier for CoverType on the GPU: a custom
  Dataset, train/valid/test loaders, a custom MLP module, and hyperparameter
  search (optionally Optuna) to reach 93%.

## 1. PyTorch is similar to NumPy is many ways, but it offers some extra features. Can you name the most important ones?

PyTorch is similar to NumPy in many ways, but it offers some extra features. The main ones are:
- Auto-differentiation
- Support for hardware accelerators
- Includes optimizers and ready-to-use neural net components

## 2. What is the difference between torch.exp() and torch.exp_(), or between torch.relu() and torch.relu_()?

torch.exp() returns a copy of the input tensor while torch.exp_() modifies it in place. Similarly, torch.relu() returns a copy while torch.relu_() modifies in place.

## 3. What are two ways to create a new tensor on the GPU?

To create a new tensor on the GPU, you can use one of the following methods:
- Set the device argument when calling torch.tensor(), torch.rand(), or other functions that create new tensors. For example, torch.randn(10, device="cuda") creates a new tensor on the CUDA GPU, with 10 random elements.
- Create a new tensor on the CPU, then transfer it to the GPU by calling its to() method, for example torch.randn(10).to("cuda"). However, it's more efficient to create the tensor directly on the GPU.
- The *_like() functions such as ones_like() and zeros_like() create a new tensor on the same device as another tensor. They also use the same data type.
- Lastly, if you execute an operation on a tensor that lives on the GPU, the result will generally be a new tensor on the GPU (unless you use an in-place operation such as torch.exp_()).

## 4. What are three ways to perform tensor computations without using autograd?

Here are three ways to perform tensor computations without using autograd:
- Manipulate tensors created with requires_grad=False (which is the default).
- Run the computations inside a with torch.no_grad(): block.
- Call the detach() method on the tensor you want to manipulate without autograd.

## 5. Will the following code cause a RuntimeError? What if you replace the second line with z = t.cos_().exp()? And what if you replace it with z = t.exp().cos_()? t = torch.tensor(2.0, requires_grad=True); z = t.cos().exp_(); z.backward() How about the following code, will it cause an error? And what if you replace the third line with w = v.cos_() * v.sin()? Will w have the same value in both cases? u = torch.tensor(2.0, requires_grad=True); v = u + 1; w = v.cos() * v.sin_(); w.backward()

Here's what happens in each case:
- The first code sample will work fine, it will not cause a RuntimeError: indeed, the cos() method creates a new (non-leaf) tensor, then exp_() modifies it in place. During the backward pass, PyTorch is able to backpropagate through the exp_() operation because the derivative of exp(x) is exp(x), so PyTorch doesn't need to know what the input x was, it can just use the output of the forward pass (i.e., exp(x)) during the backward pass.
- If you replace z = t.cos().exp_() with z = t.cos_().exp() then you will get a RuntimeError on that line ("a leaf Variable that requires grad is being used in an in-place operation"). Indeed, t is a leaf tensor (since it was created directly by the user, not the result of any computation) and you cannot apply an in-place operation on a leaf tensor with requires_grad=True.
- If you replace z = t.cos().exp_() with z = t.exp().cos_(), then you will get a RuntimeError ("one of the variables needed for gradient computation has been modified by an inplace operation") during the backward pass. Indeed, the exp() operation relies on the fact that the derivative of exp(t) is exp(t), so it doesn't need to store the tensor t for the backward pass, instead it relies on the fact that it can just use the tensor returned by t.exp() (let's call it e). So far so good. But when we call the cos_() operation, it knows that it will need its input during the backward pass (since the derivative of cos(e) is –sin(e)), so it keeps a copy of its input tensor e. Next, it tries to modify the original e in-place, and in doing so it notices that this tensor is needed by another operation (exp()) for the backward pass, so it knows that something is fishy and it raises a RuntimeError.
- The second code example will fail during backpropagation, with a RuntimeError. It's a very similar error to the previous one: the cos() operation stores a reference to its input tensor v, since v will be needed during the backward pass. Indeed, the derivative of cos(v) is –sin(v), so we need to save v. Next, the sin_() operation creates a copy of its input v (since it's an in-place operation, it knows that it must create a copy, not just preserve a reference) then it proceeds to modify the original v, but this tensor is needed to compute the gradient of cos(v), so PyTorch raises a RuntimeError.
- If you replace w = v.cos() * v.sin_() with w = v.cos_() * v.sin(), then there is no longer any RuntimeError. Indeed, the cos_() operation creates a copy of its input v so it can compute –sin(v) during the backward pass. The sin() operation is not in-place so it keeps a reference to its input v, not a copy. Backprop then runs just fine. However, there's a catch: by the time the sin() operation runs, its input v is no longer equal to 3.0, but instead it's equal to cos(3.0) since the cos_() modified v in place. As a result, w = v.cos() * v.sin_() does not give the same result as w = v.cos_() * v.sin(). The former computes cos(3) * sin(3) while the latter computes cos(3) * sin(cos(3)). And of course the gradients change as well. That's why you should be very careful with in-place operations: they can make your code faster, sure, but they can also make it silently wrong.

## 6. Suppose you create a Linear(100, 200) module. How many neurons does it have? What is the shape of is weight and bias parameters? What input shape does it expect? What output shape does it produce?

A Linear(100, 200) module has 200 neurons: one per output. Its weight tensor has a shape of [200, 100] and its bias parameter has a shape of [200]. It expects its inputs to have a shape of [..., 100], for example [32, 100], or [32, 64, 100]. It treats all dimensions independently, except for the last one. The output shape is identical to the input shape, except that the last dimension is replaced with 200. For example, if the input shape is [32, 64, 100], then the output shape is [32, 64, 200].

## 7. What are the main steps of a PyTorch training loop?

The main steps of a PyTorch training loop are:
- Prepare a batch of samples from the training set. You can use a DataLoader for this.
- Optionally transfer these samples to the GPU (typically using X_batch.to(device) and y_batch.to(device)).
- Run the inputs through the model, for example y_pred = model(X_batch).
- Compute the loss, for example loss = criterion(y_pred, y_true).
- Backpropagate through the loss using loss.backward().
- Perform an optimizer step: optimizer.step().
- Zero out the gradients: optimizer.zero_grad() (alternatively, you can do this before the backward pass, which may be safer if the gradients are non-zero before the training loop starts).
- The whole training loop is often split into epochs, but this is optional.

## 8. Why is it recommended to create the optimizer after the model is moved to the GPU?

It is recommended to create the optimizer after the model is moved to the GPU because most optimizers have some internal state, and this state is usually allocated on the same device as the model parameters.

## 9. What DataLoader options should you generally set to speed up training when using a GPU?

To speed up training when using a GPU, you should generally set the following DataLoader options:
- Set the data loader's num_workers argument to the number of processes you want to use for data loading and preprocessing. This will often speed up training by pre-fetching the next batches on the CPU while the GPU is still working on the current batch. The optimal number depends on your platform, hardware, and workload, so you should experiment with different values.
- Set the data loader's prefetch_factor argument to control the number of batches that each worker pre-fetches.
- If spawning and synchronizing workers causes too much overhead (especially on Windows), you can try setting persistent_workers=True to reuse the same workers across epochs.

## 10. What are the main classification losses provided by PyTorch, and when should you use each of them?

The main classification losses provided by PyTorch are:
- nn.CrossEntropyLoss: this is generally the loss you want to use for multiclass classification, as it's efficient and numerically stable. This loss works directly on logits, not probabilities, so your model must not include the softmax activation function on the output layer. As a result, whenever you need to estimate probabilities, you must call the F.softmax() function on the logits output by the model.
- nn.BCEWithLogitsLoss (BCE stands for binary cross-entropy): this is usually the loss you want to use for binary classification, for the same reason as nn.CrossEntropyLoss. Just like the previous loss, it works directly with logits, so your model must not include the sigmoid activation function on the output layer. Whenever you need to estimate probabilities, you must call the F.sigmoid() function on the logits output by the model. Note that some people prefer to use nn.CrossEntropyLoss even for binary classification, as it makes the code more consistent regardless of the number of classes, at a tiny computational and memory cost.
- nn.NLLLoss (NLL stands for negative log-likelihood): this is an alternative to nn.CrossEntropyLoss for multiclass classification. To use it, your model must output log probabilities rather than logits. This can be done using nn.LogSoftmax() or F.log_softmax(). This approach is a bit slower than using CrossEntropyLoss, but it can be useful if you want your model to output log probabilities rather than logits, or when you wish to tweak the probability distribution before computing the final loss (indeed, it is sometimes easier to modify log probabilities rather than logits). Whenever you need to estimate probabilities, you must call torch.exp() on the model's outputs.
- nn.BCELoss: this loss is an alternative to nn.BCEWithLogitsLoss for binary classification. It assumes that your model outputs probabilities rather than logits, so your model's output layer must use the sigmoid activation function. This can be convenient if you want your model to output probabilities directly, but it's a bit slower and less numerically stable than using BCEWithLogitsLoss.

## 11. Why is it important to call model.train() before training and model.eval() before evaluation?

Calling model.train() before training and model.eval() before evaluation is important because some layers (such as nn.Dropout, nn.BatchNorm1d or nn.BatchNorm2d) don't behave in the same way during training and evaluation, therefore we must tell the model in which mode it should run.

## 12. What is the difference between torch.jit.trace() and torch.jit.script()?

Both torch.jit.trace() and torch.jit.script() attempt to capture your model's computation graph and turn it into TorchScript code that can be optimized, saved, and deployed to various platforms. However, these functions work very differently:
- The torch.jit.trace() function runs your model with a tracing tensor that captures which operations are executed. It's quite simple and works well for simple models, but it cannot capture conditionals (e.g., if, elif, else, match): it only captures the branch of the conditional that is actually executed during tracing. Similarly, if your model contains a loop (e.g., for or while) then tracing will not capture the loop itself, it will only capture the repeated operations.
- The torch.jit.script() function actually parses your Python code to generate TorchScript code. This allows it to detect conditionals (as long as the conditions are tensors), and also capture loops. However, it only works with a subset of Python: you cannot use global variables, Python generators (yield), complex list comprehensions, variable length function arguments (*args or **kwargs), or match statements. Moreover, types must be fixed (a function cannot return an integer in some cases and a float in others), and you can only call other functions if they also respect these rules, so no standard library, no third-party libraries, etc.
