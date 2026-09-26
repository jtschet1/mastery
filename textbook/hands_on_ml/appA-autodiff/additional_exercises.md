# Appendix A — Autodiff

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The book has no exercises for this appendix, so every
card for it is here.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Differentiate f(x, y) = x²y + y + 2 by hand, evaluate both partial derivatives at (3, 4), and explain why manual differentiation doesn't scale.

Treat the other variable as a constant and apply the basic rules: the derivative of a constant is 0, the derivative of xⁿ is n·xⁿ⁻¹, the derivative of a sum is the sum of the derivatives, and constant factors carry through.

- ∂f/∂x = y·∂(x²)/∂x + 0 + 0 = 2xy, which is 2·3·4 = 24 at (3, 4).
- ∂f/∂y = x²·∂y/∂y + 1 + 0 = x² + 1, which is 9 + 1 = 10 at (3, 4).

The result is exact and reusable, but for complex functions like a deep network's loss, deriving it by hand is tedious and error-prone, and it would have to be redone every time the model changes. That's why deep learning frameworks compute gradients automatically.

## 2. How does finite difference approximation compute a gradient, and what are its drawbacks and its main practical use?

It replaces the limit in the definition of the derivative with a tiny ε (Newton's difference quotient): ∂f/∂x(x, y) ≈ (f(x + ε, y) − f(x, y))/ε, and likewise for each other input:

```python
def gradients(f, params, eps=1e-5):
    base = f(*params)
    grads = []
    for i in range(len(params)):
        tweaked = list(params)
        tweaked[i] += eps
        grads.append((f(*tweaked) - base) / eps)
    return grads
```

Drawbacks: it's approximate (for f(x, y) = x²y + y + 2 at (3, 4), ∂f/∂x comes out as about 24.00004 instead of 24), because a finite ε leaves an error while a tinier ε amplifies floating-point rounding; and it needs n + 1 evaluations of f for n parameters, which is hopeless for networks with millions of parameters. But it's trivial to implement, so it's a great way to check gradients computed by other methods: if they disagree, the other method is probably buggy.

## 3. Walk through forward-mode autodiff on the computation graph of g(x, y) = 5 + xy to get ∂g/∂x. What kind of result does it produce?

It traverses the graph from the inputs to the output, building each node's derivative from the derivatives of its inputs:

1. Leaves: the constant 5 gives 0, x gives ∂x/∂x = 1, and y gives ∂y/∂x = 0.
2. Product node x·y: the product rule ∂(u·v)/∂x = u·∂v/∂x + v·∂u/∂x gives x·0 + y·1.
3. Sum node: the derivative of a sum is the sum of the derivatives, so ∂g/∂x = 0 + (x·0 + y·1).

After pruning the useless operations, this simplifies to ∂g/∂x = y. The output is itself a computation graph (symbolic differentiation): you can evaluate it for any x and y, and run forward-mode autodiff on it again to get higher-order derivatives. The downside is that for complex functions the derivative graph can become huge and hard to simplify, which hurts performance.

## 4. What are dual numbers, and how do they give you both f(3, 4) and ∂f/∂x(3, 4) in one pass for f(x, y) = x²y + y + 2?

A dual number is a + bε, where ε is an infinitesimal such that ε² = 0 (though ε ≠ 0); in memory it's just the pair (a, b). The arithmetic follows from ε² = 0: (a + bε) + (c + dε) = (a + c) + (b + d)ε, and (a + bε)·(c + dε) = ac + (ad + bc)ε. The key property is h(a + bε) = h(a) + b·h′(a)·ε, so evaluating h(a + ε) yields both h(a) and h′(a).

For ∂f/∂x at (3, 4), evaluate f(3 + ε, 4):
- x² = (3 + ε)² = 9 + 6ε
- x²y = (9 + 6ε)·4 = 36 + 24ε
- x²y + y + 2 = 42 + 24ε

So f(3, 4) = 42 and ∂f/∂x(3, 4) = 24. This is forward-mode autodiff computed numerically, without building a derivative graph. ∂f/∂y needs a second pass: f(3, 4 + ε) = 42 + 10ε.

## 5. Sketch a minimal dual-number class in Python and use it to compute ∂f/∂x and ∂f/∂y for f(x, y) = x·x·y + y + 2 at (3, 4).

Store the value and the ε coefficient, implement the dual-number rules for addition and multiplication, and promote plain numbers to dual numbers with an ε part of 0:

```python
class Dual:
    def __init__(self, value, eps=0.0):
        self.value, self.eps = value, eps
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value + o.value, self.eps + o.eps)
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value * o.value, self.value * o.eps + self.eps * o.value)
    __radd__, __rmul__ = __add__, __mul__

f = lambda x, y: x * x * y + y + 2
```

To differentiate with respect to one input, give it an ε part of 1 and the others 0: f(Dual(3, 1), Dual(4)) has value 42 and eps 24.0, which is ∂f/∂x, and f(Dual(3), Dual(4, 1)).eps is 10.0, which is ∂f/∂y. Each input needs its own pass.

## 6. How many passes or function evaluations do finite differences, forward-mode, and reverse-mode autodiff need for the gradient of a loss with n parameters, and why?

- Finite differences: n + 1 evaluations of the function, and the result is only approximate.
- Forward mode: n passes, because each pass (seeding one input with ε) yields the derivatives with respect to a single input. Exact, but 1,000 parameters means 1,000 passes.
- Reverse mode: one forward pass plus one reverse pass per output, whatever n is. A loss is a single output, so two passes give the whole gradient.

The chain rule explains the asymmetry: ∂z/∂x = ∂s₁/∂x · ∂s₂/∂s₁ · … · ∂z/∂sₙ. Forward mode multiplies these factors starting from the input side, reverse mode from the output side, so reverse mode computes the output-side factors once and shares them across all inputs. With few inputs and many outputs, forward mode is cheaper; neural nets have many inputs (the parameters) and one output (the loss), hence reverse mode.

## 7. Walk through reverse-mode autodiff by hand for f(x, y) = x²y + y + 2 at x = 3, y = 4.

Name the intermediate nodes: a = x·x, b = a·y, c = y + 2, and f = b + c.

Forward pass (compute and keep every node's value): a = 9, b = 36, c = 6, f = 42.

Reverse pass, from the output down, applying the chain rule ∂f/∂node = ∂f/∂parent · ∂parent/∂node:
- ∂f/∂f = 1
- ∂f/∂b = 1 and ∂f/∂c = 1, since a sum passes the gradient through unchanged
- ∂f/∂a = ∂f/∂b · y = 4
- ∂f/∂y = ∂f/∂b · a + ∂f/∂c · 1 = 9 + 1 = 10, because y feeds two nodes and their contributions add up
- ∂f/∂x = ∂f/∂a · 2x = 4 · 6 = 24

These two passes give every partial derivative, however many inputs there are. Notice that the reverse pass reused values stored during the forward pass (x, y, and a).

## 8. How does PyTorch's autograd map onto reverse-mode autodiff: what plays the role of the graph, the forward pass, and the reverse pass?

- The graph: built on the fly as operations run. Every tensor computed from tensors with requires_grad=True gets a grad_fn attribute, an operation-specific backward node (e.g., MulBackward0) linked to the nodes of its inputs. Gradients are computed for the leaves created with requires_grad=True (typically the parameters).
- The forward pass: simply running your code. Each backward node also saves whatever its local derivative will need (e.g., both factors of a product).
- The reverse pass: loss.backward() walks the graph from the loss back to the leaves, applying the chain rule at each node, summing contributions when a tensor was used several times, and adding the results into each leaf's .grad attribute.

A fresh graph is recorded on every forward pass, containing only the operations that actually ran, so Python loops and conditionals just work. Autodiff also copes with functions that aren't differentiable everywhere, as long as they're differentiable at the points where you evaluate them.

## 9. How do you compute second-order derivatives with PyTorch's autograd?

Use torch.autograd.grad(), which returns the gradients instead of accumulating them into .grad, with create_graph=True so that the gradient computation is itself recorded as a graph you can differentiate again:

```python
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
f = x * x * y + y + 2
dfdx, dfdy = torch.autograd.grad(f, [x, y], create_graph=True)  # 2xy = 24, x² + 1 = 10
d2f_dx2, d2f_dxdy = torch.autograd.grad(dfdx, [x, y])            # 2y = 8, 2x = 6
```

Two gotchas: by default a graph is freed once it's been backpropagated through, so pass retain_graph=True if you need to go through the same graph again; and if an input doesn't affect the output (e.g., differentiating ∂f/∂y = x² + 1 with respect to y), grad() raises an error unless you pass allow_unused=True, in which case it returns None for that input.

## 10. Why does reverse-mode autodiff make training use much more memory than inference, and how can you trade computation for memory?

The reverse pass needs values from the forward pass: for instance, the derivative of u·v with respect to u is v. So every intermediate result that a backward step depends on (such as each layer's activations) must be kept until the reverse pass has used it, whereas inference can discard each activation once the next layer has consumed it. That's also why inference should run under torch.no_grad() or torch.inference_mode(): no graph is recorded, so nothing extra is kept.

With deep networks and large inputs, these stored values can exhaust GPU memory. Besides shrinking the model or the batch, you can offload them to CPU RAM, or keep only some of them (e.g., every other layer's) and recompute the missing ones from the nearest stored value during the reverse pass. This gradient (or activation) checkpointing trades extra computation for memory; PyTorch provides it in torch.utils.checkpoint.
