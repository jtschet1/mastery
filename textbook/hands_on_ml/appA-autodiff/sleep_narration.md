# Appendix A — Autodiff, to fall asleep to

Appendix A. Autodiff.

---

Computing partial derivatives.

Automatic differentiation, or autodiff, is how PyTorch computes gradients. The appendix explains how it works and how it compares with the alternatives.

The running example is a function f of x and y, equal to x squared times y, plus y, plus two. Gradient descent and other optimization algorithms need its partial derivatives with respect to x and with respect to y. There are four main ways to get them: manual differentiation, finite difference approximation, forward-mode autodiff, and reverse-mode autodiff. PyTorch implements reverse-mode autodiff, which is easiest to understand after the other three.

---

Manual differentiation.

Manual differentiation means deriving the derivatives with pencil, paper, and calculus. For f, five rules are enough. The derivative of a constant is zero. The derivative of a constant lambda times x is lambda. The derivative of x to the power lambda is lambda times x to the power lambda minus one, so the derivative of x squared is two x. The derivative of a sum of functions is the sum of their derivatives. The derivative of lambda times a function is lambda times the function's derivative.

The partial derivative of f with respect to x treats y as a constant. The first term, x squared times y, gives two x times y. The second term, y, and the third term, two, are constants with respect to x, so they give zero. The result is two x y.

The partial derivative of f with respect to y treats x as a constant. The first term gives x squared, the second gives one, and the third gives zero. The result is x squared plus one.

At x equal to three and y equal to four, these give twenty-four and ten.

This approach becomes very tedious for complex functions, and it is easy to make mistakes.

---

Finite difference approximation.

The derivative of a function h at a point x zero is the slope of the function at that point. It is defined as the limit of the slope of a straight line through the point x zero and another point x on the function, as x gets infinitely close to x zero. Equivalently, it is the limit, as epsilon goes to zero, of h at x zero plus epsilon, minus h at x zero, all divided by epsilon.

To approximate the partial derivative of f with respect to x at x equal to three and y equal to four, the function is evaluated at three plus a very small epsilon and four, the value at three and four is subtracted, and the difference is divided by epsilon. This kind of numerical approximation is called a finite difference approximation, and this specific formula is called Newton's difference quotient.

With epsilon equal to one hundred-thousandth, the results are imprecise: about twenty-four point zero zero zero zero four instead of exactly twenty-four, and slightly more than ten instead of exactly ten. The error gets worse for more complicated functions.

Computing both partial derivatives requires calling f at least three times: once at the point itself, and once with each input nudged. With one thousand parameters, f must be called at least one thousand and one times. For large neural networks, this makes finite difference approximation far too inefficient.

The method is so simple to implement that it is a good way to check that other methods are implemented correctly. If it disagrees with a manually derived derivative, the manual derivation probably contains a mistake.

Manual differentiation and finite difference approximation are both fatally flawed for training large-scale neural networks.

---

Forward-mode autodiff.

Forward-mode autodiff is easiest to see on an even simpler function, g of x and y, equal to five plus x times y. Its computation graph has a constant node for five, variable nodes for x and y, a multiplication node, and an addition node.

The algorithm goes through the computation graph from the inputs to the output, which is why it is called forward mode. To get the partial derivative with respect to x, it starts with the leaf nodes. The constant node returns zero, since the derivative of a constant is always zero. The variable x returns one, since the derivative of x with respect to x is one. The variable y returns zero, since the derivative of y with respect to x is zero. For the partial derivative with respect to y, the roles of x and y would be reversed.

Next comes the multiplication node. By the product rule, the derivative of u times v is the derivative of v times u, plus v times the derivative of u. This builds the part of a new graph that represents zero times x, plus y times one.

Last comes the addition node. The derivative of a sum is the sum of the derivatives, so an addition node is created and connected to the parts already built. The result represents zero, plus zero times x, plus y times one, which is the correct partial derivative.

That graph can be simplified a lot. Pruning the unnecessary operations leaves a single node: the partial derivative of g with respect to x is simply y. Simplification is easy here, but for a more complex function, forward-mode autodiff can produce a huge graph that is hard to simplify and leads to suboptimal performance.

Starting from a computation graph and producing another computation graph is called symbolic differentiation. It has two useful features. First, once the derivative's graph exists, it can be reused to compute the derivative for any values of x and y. Second, running forward-mode autodiff again on the resulting graph gives second-order derivatives, meaning derivatives of derivatives, and the process can be repeated for third-order derivatives and beyond.

---

Dual numbers.

Forward-mode autodiff can also run without building a graph, numerically rather than symbolically, by computing intermediate results on the fly. One way to do this uses dual numbers.

A dual number has the form a plus b epsilon, where a and b are real numbers and epsilon is an infinitesimal number whose square is zero, although epsilon itself is not zero. Loosely, the dual number forty-two plus twenty-four epsilon is like forty-two point zero zero zero and so on, with infinitely many zeros, followed by twenty-four. In memory, a dual number is stored as a pair of floats, so forty-two plus twenty-four epsilon is stored as the pair forty-two and twenty-four.

Dual numbers can be added, multiplied, and so on. Lambda times a plus b epsilon is lambda a plus lambda b epsilon. The sum of a plus b epsilon and c plus d epsilon is a plus c, plus b plus d times epsilon. Their product is a c, plus a d plus b c times epsilon, plus b d times epsilon squared; since epsilon squared is zero, the last term disappears, leaving a c plus a d plus b c times epsilon.

The key property is that h of a plus b epsilon equals h of a, plus b times the derivative of h at a, times epsilon. So evaluating h at a plus epsilon gives both h of a and the derivative of h at a in one shot.

For f at x equal to three and y equal to four, evaluating f at three plus epsilon and four returns a dual number. Its first component is f at three and four, which is forty-two, and its second component is the partial derivative of f with respect to x at that point, which is twenty-four. The partial derivative with respect to y requires a second pass through the graph, this time with x equal to three and y equal to four plus epsilon.

Forward-mode autodiff is much more accurate than finite difference approximation, but it shares the same major flaw when there are many inputs and few outputs, as with neural networks. With one thousand parameters, it needs one thousand passes through the graph to compute all the partial derivatives.

---

Reverse-mode autodiff.

Reverse-mode autodiff is the solution implemented by PyTorch. It first goes through the graph in the forward direction, from the inputs to the output, computing the value of each node. Then it makes a second pass in the reverse direction, from the output to the inputs, computing all the partial derivatives. The name comes from this second pass, in which gradients flow backward through the graph.

In the book's example, the forward pass for f starts from x equal to three and y equal to four, and the nodes are labeled n one to n seven. Node n two holds y, which is four. Node n four is x times x, which is nine. Node n five is n four times n two. Node n seven, the output, is the sum of n five and n six, and its value is forty-two, which is f at three and four.

The reverse pass moves down the graph, computing the partial derivative of f with respect to each node in turn until it reaches the variable nodes. It relies on the chain rule: the partial derivative of f with respect to x equals the partial derivative of f with respect to an intermediate node, times the partial derivative of that node with respect to x.

Since n seven is the output node, f equals n seven, and the derivative of f with respect to n seven is one.

Next is n five. The derivative of f with respect to n five is the derivative of f with respect to n seven, times the derivative of n seven with respect to n five. N seven is the sum of n five and n six, so the derivative of n seven with respect to n five is one. The derivative of f with respect to n five is one times one, which is one.

Next is n four. The derivative of f with respect to n four is the derivative of f with respect to n five, times the derivative of n five with respect to n four. N five is n four times n two, so the derivative of n five with respect to n four is n two, which is four. The derivative of f with respect to n four is one times four, which is four.

The process continues until it reaches the bottom of the graph. At that point, all the partial derivatives of f at x equal to three and y equal to four are known: twenty-four with respect to x and ten with respect to y, matching the manual results.

Reverse-mode autodiff is powerful and accurate, especially when there are many inputs and few outputs. It needs only one forward pass plus one reverse pass per output to compute the partial derivatives of every output with respect to every input. Training a neural network minimizes a single output, the loss, so only two passes through the graph are needed to compute all the gradients.

---

Autodiff in PyTorch.

PyTorch builds a new graph on the fly during each forward pass. Whenever an operation runs on a tensor that requires gradients, PyTorch computes the resulting tensor and sets its grad function attribute to an operation-specific object. That object knows how to propagate gradients backward through the operation. Because the graph is rebuilt on the fly every time, the code can be highly dynamic, with loops and conditionals, and gradients still work.

Reverse-mode autodiff can also handle functions that are not differentiable everywhere, as long as the partial derivatives are requested at points where the function is differentiable.

Building a tiny autodiff framework from scratch, for a small set of operations, is a good exercise for truly mastering autodiff. The book's repository includes an extra notebook that does this, and Andrej Karpathy has a video in which he builds the micrograd library from scratch.

---

Summary.

Gradients can be computed in four main ways. Manual differentiation is tedious and error-prone. Finite difference approximation is imprecise and needs at least one extra function call per parameter, but it is a simple way to check other methods.

Forward-mode autodiff goes through the graph from the inputs to the output. It can work symbolically, producing a new graph for the derivative that can be reused and differentiated again, or numerically, with dual numbers, where evaluating a function at a plus epsilon yields both its value and its derivative. It is accurate, but it needs one pass per input, which is too slow for networks with many parameters.

Reverse-mode autodiff makes a forward pass to compute every node's value, then a reverse pass that applies the chain rule from the output down to the inputs. It needs one reverse pass per output, so with a single loss, all the gradients cost just two passes through the graph.

PyTorch implements reverse-mode autodiff and builds the graph on the fly during each forward pass, which allows fully dynamic code with loops and conditionals.
