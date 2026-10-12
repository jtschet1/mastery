# Chapter 7 — Dimensionality Reduction, to fall asleep to

Chapter seven. Dimensionality reduction.

---

Many machine learning problems have thousands or even millions of features per training instance. So many features make training very slow, and they can make a good solution much harder to find. This problem is called the curse of dimensionality.

In real-world problems the number of features can often be reduced considerably, which can turn an intractable problem into a tractable one. In the MNIST images of handwritten digits, the pixels near the borders are almost always white, so they can be dropped from the training set with little loss of information. The previous chapter showed that these pixels are unimportant for classification. Neighboring pixels are also often highly correlated, so merging two of them into one, for example by averaging their intensities, loses little information and removes redundancy, and sometimes noise.

Reducing dimensionality does drop some information, much as compressing an image to JPEG degrades its quality. It can make a system perform slightly worse, especially if dimensionality is reduced too much. Some models, such as neural networks, handle high-dimensional data efficiently and learn to reduce its dimensionality themselves while keeping what matters for the task. So an extra dimensionality reduction step before training does not always help.

Besides speeding up training and possibly improving performance, dimensionality reduction is useful for data visualization. Reducing a dataset to two or three dimensions allows a condensed view of it to be plotted, which can reveal patterns such as clusters. Visualization is also essential for communicating conclusions to people who are not data scientists, in particular the decision makers who will use the results.

---

The curse of dimensionality. Intuition built in three dimensions fails in high-dimensional space. Even a four-dimensional hypercube, called a tesseract, is very hard to picture.

Many things behave differently in high dimensions. A random point in a unit square, a square with sides of length one, has only about a zero point four percent chance of lying within one thousandth of a border. It is very unlikely to be extreme along any dimension. In a ten-thousand-dimensional unit hypercube, that probability is greater than ninety-nine point nine nine nine nine nine nine percent. Most points in a high-dimensional hypercube are very close to the border.

Distances behave strangely too. Two random points in a unit square are on average about zero point five two apart. In a three-dimensional unit cube, the average is about zero point six six. In a one-million-dimensional unit hypercube, the average distance is about four hundred and eight, roughly the square root of one sixth of a million. Two points can be that far apart inside the same unit hypercube because high-dimensional space has so much room.

As a result, high-dimensional datasets are often very sparse: most training instances are far from one another. Methods based on distance or similarity, such as k-nearest neighbors, become much less effective. Some models, such as support vector machines and dense neural networks, scale poorly with dimensionality and may not be usable at all. New instances are likely to be far from any training instance, so predictions rest on much larger extrapolations and are less reliable. Patterns are harder to identify, so models fit the noise more often, and regularization becomes even more important. Models also become harder to interpret.

In theory, the training set could be enlarged until instances are dense enough. In practice, the number of instances needed to reach a given density grows exponentially with the number of dimensions. With just one hundred features, each ranging from zero to one, and instances spread uniformly, more training instances than atoms in the observable universe would be needed for instances to lie within zero point one of each other on average. MNIST has far more than one hundred features.

---

Projection. Projection and manifold learning are the two main approaches to dimensionality reduction.

In most real-world problems, training instances are not spread uniformly across all dimensions. Many features are almost constant, and others are highly correlated. As a result, all training instances lie within, or close to, a much lower-dimensional subspace of the high-dimensional space.

The book's example is a three-dimensional dataset whose instances all lie close to a plane, which is a two-dimensional subspace. Projecting every instance perpendicularly onto that plane gives a new two-dimensional dataset. Its two new features, called z one and z two, are the coordinates of the projections on the plane.

---

Manifold learning. Projection is fast and often works well, but the subspace may twist and turn. The Swiss roll is a toy dataset of three-dimensional points in the shape of a rolled-up cake. Projecting it onto a plane, for example by dropping the third coordinate, squashes different layers of the roll together. What is wanted instead is to unroll it into a flat two-dimensional dataset.

The Swiss roll is a two-dimensional manifold: a two-dimensional shape bent and twisted in a higher-dimensional space. More generally, a d-dimensional manifold is a part of an n-dimensional space, with d smaller than n, that locally resembles a d-dimensional hyperplane. The Swiss roll locally resembles a flat plane, but it is rolled up in the third dimension.

Many dimensionality reduction algorithms, such as locally linear embedding, Isomap, t-SNE, and UMAP, work by modeling the manifold the training instances lie on. This is called manifold learning. It relies on the manifold assumption, also called the manifold hypothesis: most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This is very often observed empirically.

MNIST illustrates it. Handwritten digit images share structure: they are made of connected lines, their borders are white, and they are roughly centered. Among randomly generated images, only a tiny fraction would look like handwritten digits. The degrees of freedom available when creating a digit image are far fewer than when generating any image at all, and those constraints squeeze the dataset into a lower-dimensional manifold.

The manifold assumption often comes with a second, implicit assumption: that the task, such as classification or regression, is simpler in the lower-dimensional space of the manifold. Sometimes it is. A Swiss roll split into two classes can need a complex decision boundary in three dimensions, but only a straight line once unrolled. Sometimes it is not. A decision boundary where the first coordinate equals five is a simple vertical plane in the original space, but on the unrolled manifold it becomes four separate line segments.

So reducing dimensionality before training usually speeds training up, but it does not always give a better or simpler solution; that depends on the dataset. It tends to help most when the dataset is small relative to the number of features, especially when the data is noisy or many features are highly correlated, meaning redundant. If domain knowledge says the process that generated the data is simple, the manifold assumption holds, and dimensionality reduction is likely to help.

---

Principal component analysis. PCA is by far the most popular dimensionality reduction algorithm. It finds the hyperplane that lies closest to the data, then projects the data onto it.

Before projecting, the right hyperplane must be chosen. For a simple two-dimensional dataset and three candidate axes, projecting onto one axis keeps most of the variance, projecting onto another keeps very little, and the third is in between. The axis that preserves the most variance is the reasonable choice, because it probably loses the least information. A shadow cast on the ground by the sun directly overhead is a small blob that looks nothing like the person. A shadow on a wall at sunrise is larger, and it does look like them.

The axis that maximizes the variance is also the axis that minimizes the mean squared distance between the original data and its projection onto that axis. This is the simple idea behind PCA, which Karl Pearson introduced in 1901.

---

Principal components. PCA finds the axis that accounts for the largest amount of variance in the training set. It then finds a second axis, orthogonal to the first, that accounts for the largest share of the remaining variance. In a higher-dimensional dataset it continues with a third axis orthogonal to both, then a fourth, a fifth, and so on, finding as many axes as there are dimensions. The i-th axis is called the i-th principal component, or PC, of the data. For the three-dimensional dataset lying near a plane, the first two principal components lie in the plane, and the third is perpendicular to it.

For each principal component, PCA finds a zero-centered unit vector pointing along it. The direction of that vector is not guaranteed. If the training set is perturbed slightly and PCA is run again, a unit vector may point the opposite way, and two vectors may even rotate or swap if the variances along their axes are very close. So when PCA is a preprocessing step before a model, the model must be retrained entirely every time the PCA transformer is updated. Otherwise the new PCA output may not line up with the old one, and the model will be confused.

The principal components are found with singular value decomposition, or SVD, a standard matrix factorization technique. It decomposes the training set matrix X into the product of three matrices: U, sigma, and the transpose of V. The matrix V contains the unit vectors that define all the principal components, in the correct order. NumPy's linear algebra module has an SVD function that returns these matrices, and the first rows of the transposed V matrix are the first principal components.

PCA assumes the dataset is centered around the origin. Scikit-Learn's PCA classes center the data automatically. A hand-written implementation, or another library, may require subtracting each feature's mean first.

---

Projecting down to d dimensions. Once the principal components are known, the dataset can be reduced to d dimensions by projecting it onto the hyperplane defined by the first d principal components. This choice of hyperplane preserves as much variance as possible. The three-dimensional example, projected onto the plane of its first two components, keeps a large part of its variance and looks very much like the original.

The projection is a single matrix multiplication: the training set matrix times a matrix made of the first d columns of V. The result is the reduced dataset, with d dimensions.

Scikit-Learn's PCA class uses SVD in the same way and centers the data automatically. Its n components hyperparameter sets the number of dimensions to keep, and its fit transform method fits the model and reduces the data in one step. After fitting, its components attribute holds one row for each of the first d principal components, which is the transpose of the projection matrix.

---

Explained variance ratio. For each principal component, the explained variance ratio gives the proportion of the dataset's variance that lies along it. Scikit-Learn stores it in an attribute after fitting. For the three-dimensional example, about eighty-two percent of the variance lies along the first component and about eleven percent along the second. That leaves about seven percent for the third, which probably carries little information.

---

Choosing the right number of dimensions. Rather than picking an arbitrary number, it is simpler to keep enough dimensions for their explained variance to add up to a sufficiently large share, such as ninety-five percent. The exception is visualization, which needs two or three dimensions.

One way is to run PCA without reducing dimensionality, compute the cumulative sum of the explained variance ratios, and find the first number of dimensions at which it reaches ninety-five percent. For the MNIST training set, which has seven hundred and eighty-four features, that number is one hundred and fifty-four. A simpler option is to set n components to a fraction between zero and one, the ratio of variance to preserve, such as zero point nine five. The actual number of components is then determined during training and stored in an attribute.

Another option is to plot the cumulative explained variance against the number of dimensions. The curve usually has an elbow, where the explained variance stops growing fast. For MNIST, reducing to about one hundred dimensions would not lose much explained variance.

When dimensionality reduction is a preprocessing step for a supervised task such as classification, the number of dimensions can be tuned like any other hyperparameter. The book's example builds a two-step pipeline, PCA followed by a random forest classifier, and runs a randomized search over the number of PCA components and the number of trees in the forest. It is a quick search: ten iterations, three-fold cross-validation, and only one thousand training instances. The best combination keeps just fifty-seven components out of seven hundred and eighty-four, with four hundred and seventy-five trees. The number is low because a random forest is a powerful model. A linear model, such as a stochastic gradient descent classifier, would need to keep more dimensions, about seventy-five.

Model size and speed may matter as well as performance. Fewer dimensions mean a smaller model and faster training and inference. Shrinking the data too much loses too much signal, and the model underfits. The right balance of speed, size, and performance depends on the use case.

---

PCA for compression. After dimensionality reduction, the training set takes much less space. MNIST reduced with PCA while preserving ninety-five percent of its variance has one hundred and fifty-four features instead of seven hundred and eighty-four. That is less than twenty percent of its original size, for a loss of only five percent of its variance. Such a size reduction can speed up a classification algorithm tremendously.

The reduced dataset can be decompressed back to seven hundred and eighty-four dimensions by applying the inverse of the PCA projection, which multiplies the reduced data by the transpose of the projection matrix. Scikit-Learn provides it as the inverse transform method. This does not recover the original data exactly, because the projection lost the information in the dropped five percent of variance, but the result is close. Decompressed MNIST digits show a slight loss of image quality but remain mostly intact.

The mean squared distance between the original data and the reconstructed data, meaning the data after compression and decompression, is called the reconstruction error.

---

Randomized PCA. Setting the svd solver hyperparameter to randomized makes Scikit-Learn use a stochastic algorithm that quickly finds an approximation of the first d principal components. Its computational complexity is on the order of m times d squared, plus d cubed, where m is the number of instances. Full SVD is on the order of m times n squared, plus n cubed, where n is the number of features. So randomized PCA is dramatically faster when d is much smaller than n.

By default the solver is set to auto. If the data has fewer than one thousand features and more than ten times as many samples as features, Scikit-Learn uses a solver based on the eigendecomposition of the covariance matrix, which is very fast in those conditions. Otherwise, if the larger of m and n exceeds five hundred and the number of components is an integer below eighty percent of the smaller of m and n, it uses the randomized solver. In other cases it uses full SVD. Setting the solver to full forces full SVD, trading compute time for a slightly more precise result.

---

Incremental PCA. The implementations so far need the whole training set to fit in memory. Incremental PCA, or IPCA, algorithms split the training set into mini-batches and feed them in one at a time. This suits large training sets, and applying PCA online, meaning on the fly as new instances arrive.

The book splits the MNIST training set into one hundred mini-batches with NumPy's array split function and feeds them one by one to Scikit-Learn's incremental PCA class, reducing to one hundred and fifty-four dimensions as before. Each mini-batch goes to the partial fit method, not the fit method.

An alternative is NumPy's memory map class, which handles a large array stored in a binary file on disk as if it were entirely in memory, loading only the data it needs when it needs it. The training set is written to a memory-mapped file, chunk by chunk when it does not fit in memory, and flushed to disk. The file is then loaded and passed to the incremental PCA class along with a batch size. Because the class uses only a small part of the array at any time, memory use stays under control, and the ordinary fit method can be called. Only raw binary data is saved to disk, so the data type and shape must be specified when loading; without the shape, the result is a one-dimensional array.

For very high-dimensional datasets, PCA can still be too slow. Even randomized PCA costs on the order of m times d squared plus d cubed, so the target dimension d must not be too large. With tens of thousands of features or more, as with images, random projection is worth considering instead.

---

Random projection. The random projection algorithm projects the data to a lower-dimensional space using a random linear projection. Such a projection is very likely to preserve distances fairly well, as William B. Johnson and Joram Lindenstrauss showed in a famous lemma. Similar instances stay similar after the projection, and very different instances stay very different.

The more dimensions are dropped, the more information is lost and the more distances get distorted. Johnson and Lindenstrauss gave an equation for the minimum number of dimensions that ensures, with high probability, that distances change by no more than a given tolerance, called epsilon. The target dimension d must be at least four times the natural logarithm of m, divided by the quantity one half epsilon squared minus one third epsilon cubed, where m is the number of instances. With five thousand instances of twenty thousand features each, and squared distances allowed to change by at most ten percent, d is seven thousand three hundred. The equation depends only on m and epsilon, not on the number of features. Scikit-Learn implements it as the Johnson Lindenstrauss min dim function.

The projection uses a random matrix with d rows and n columns, each entry sampled from a Gaussian distribution with mean zero and variance one over d. Multiplying the dataset by the transpose of this matrix projects it from n dimensions down to d. Training is almost instantaneous: creating the random matrix needs only the dataset's shape, and the data itself is not used at all. This makes random projection well suited to very high-dimensional data, such as text or genomics with millions of features, and to very sparse data, where even randomized PCA may take too long to train and need too much memory. At inference time, random projection is as fast as PCA, a single matrix multiplication. It is not a silver bullet: it loses a bit more signal than PCA, so it trades performance for training speed.

Scikit-Learn's Gaussian random projection class does exactly this. Its fit method uses the Johnson and Lindenstrauss equation to determine the output dimensionality, then generates the random matrix and stores it in its components attribute. Its transform method performs the projection. An epsilon hyperparameter sets the tolerance, with a default of zero point one, and the n components hyperparameter forces a specific target dimensionality. Both can be fine-tuned with cross-validation.

Scikit-Learn's sparse random projection class determines the target dimensionality the same way, generates a random matrix of the same shape, and performs the projection identically, but its random matrix is sparse. In the book's example it uses about twenty-five megabytes of memory instead of almost one point two gigabytes, and it is about fifty percent faster, both to generate the matrix and to reduce dimensionality. Sparse input stays sparse, unless dense output is requested. It has the same distance-preserving property, and its quality is comparable, only very slightly less accurate. It is usually preferable to the Gaussian version, especially for large or sparse datasets.

The fraction of nonzero entries in the sparse random matrix, called r, is its density. By default it is one over the square root of the number of features. With twenty thousand features, only about one cell in one hundred and forty-one is nonzero. The density hyperparameter can set another value. Each cell is nonzero with probability r, and each nonzero value is either plus v or minus v with equal probability, where v is one over the square root of d times r.

To invert a random projection, compute the pseudoinverse of the components matrix, then multiply the reduced data by the transpose of that pseudoinverse. This can take a very long time for a large components matrix: computing the pseudoinverse costs on the order of d times n squared if d is smaller than n, and n times d squared otherwise.

Random projection is a simple, fast, memory-efficient, and surprisingly powerful dimensionality reduction algorithm, especially for high-dimensional datasets.

It is not only used to reduce the dimensionality of large datasets. A 2017 paper by Sanjoy Dasgupta and colleagues showed that the brain of a fruit fly implements an analog of random projection. It maps dense, low-dimensional olfactory inputs to sparse, high-dimensional binary outputs. For each odor only a small fraction of the output neurons activate, but similar odors activate many of the same neurons. This resembles locality sensitive hashing, or LSH, an algorithm search engines typically use to group similar documents.

---

Locally linear embedding. LLE is a nonlinear dimensionality reduction technique, and a manifold learning technique. Unlike PCA and random projection, it does not rely on projections. It first determines how each training instance linearly relates to its nearest neighbors, then looks for a low-dimensional representation of the training set that best preserves those local relationships. This makes it particularly good at unrolling twisted manifolds, especially when there is not too much noise. It does not scale well, so it is mostly for small or medium-sized datasets.

The book generates a Swiss roll of one thousand points with Scikit-Learn's make swiss roll function, then unrolls it with the locally linear embedding class, using two components and ten neighbors. The function also returns each instance's position along the rolled axis, which could serve as the target of a nonlinear regression task. The roll comes out completely unrolled, and distances between instances are well preserved locally. On a larger scale they are not: the unrolled Swiss roll should be a rectangle, but it comes out as a stretched and twisted band. LLE still models the manifold well.

LLE works in two steps. In the first, for each training instance, the algorithm finds its k nearest neighbors, ten in the example, and tries to reconstruct the instance as a weighted sum of those neighbors. It finds the weights that minimize the squared distance between each instance and its reconstruction, summed over all instances. Two constraints apply: the weight is zero for any instance that is not one of the k nearest neighbors, and each instance's weights add up to one, which normalizes them. The resulting weight matrix encodes the local linear relationships between the training instances.

In the second step, each training instance is mapped to a point in a d-dimensional space, with d smaller than n, so that these local relationships are preserved as much as possible. The algorithm minimizes the squared distance between each mapped point and the same weighted sum of its neighbors' mapped points, summed over all instances. This mirrors the first step. Instead of keeping the instances fixed and finding the best weights, it keeps the weights fixed and finds the best positions of the instances' images in the low-dimensional space.

Scikit-Learn's implementation costs on the order of m log m times n log k to find the nearest neighbors, m times n times k cubed to optimize the weights, and d times m squared to build the low-dimensional representations. The m squared term makes it scale poorly to very large datasets. LLE is quite different from the projection techniques and significantly more complex, but it can build much better low-dimensional representations, especially when the data is nonlinear.

---

Other dimensionality reduction techniques available in Scikit-Learn.

Multidimensional scaling, or MDS, reduces dimensionality while trying to preserve the distances between instances. Random projection does that for high-dimensional data, but it does not work well on low-dimensional data.

Isomap connects each instance to its nearest neighbors to build a graph, then reduces dimensionality while trying to preserve the geodesic distances between instances. The geodesic distance between two nodes in a graph is the number of nodes on the shortest path between them. Isomap works best when the data lies on a fairly smooth, low-dimensional manifold with a single global structure, such as the Swiss roll.

t-distributed stochastic neighbor embedding, or t-SNE, reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space, such as a two-dimensional map of the MNIST images. It is not meant as a preprocessing step for a machine learning model.

Linear discriminant analysis, or LDA, is a linear classification algorithm that learns the most discriminative axes between the classes during training. These axes can define a hyperplane onto which the data is projected. The projection keeps the classes as far apart as possible, so LDA is a good way to reduce dimensionality before running another classification algorithm, unless LDA alone is sufficient.

Uniform manifold approximation and projection, or UMAP, is another popular technique for visualization. t-SNE is better at preserving local structure, especially clusters, while UMAP tries to preserve both local and global structure, and it scales better to large datasets. It is not in Scikit-Learn, but the umap-learn package has a good implementation.

On the Swiss roll, MDS flattens the roll without losing its global curvature. Isomap drops the curvature entirely. t-SNE flattens it reasonably while keeping a bit of curvature, and it also amplifies clusters, tearing the roll apart. Whether preserving large-scale structure, or amplifying clusters, is good or bad depends on the downstream task.

---

Summary.

High-dimensional data suffers from the curse of dimensionality. Random points lie far apart and close to the borders of the space, datasets are sparse, distance-based methods weaken, models overfit more easily, and the data needed to fill the space grows exponentially with the number of dimensions.

Dimensionality reduction speeds up training, can improve performance, enables visualization, and compresses data, but it loses some information and does not always help. Its two main approaches are projection onto a lower-dimensional subspace, and manifold learning, which models a twisted lower-dimensional shape. Manifold learning relies on the manifold assumption: real high-dimensional data lies near a much lower-dimensional manifold.

PCA projects data onto the hyperplane that preserves the most variance, spanned by the principal components that SVD finds. The number of dimensions is usually chosen to keep a target share of the variance, such as ninety-five percent, or tuned as a hyperparameter. Randomized PCA is faster when the target dimension is small, incremental PCA handles data that does not fit in memory, and the inverse transformation reconstructs the data with some reconstruction error.

Random projection multiplies the data by a random matrix, sized with the Johnson and Lindenstrauss equation, and preserves distances approximately. It trains almost instantly and suits very high-dimensional or sparse data; the sparse version is usually preferred.

LLE is a nonlinear manifold learning method that preserves each instance's linear relationships with its nearest neighbors, and it works well on twisted manifolds of small or medium size. MDS, Isomap, t-SNE, UMAP, and LDA are other options, with t-SNE and UMAP used mainly for visualization.
