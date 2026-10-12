# Chapter 8 — Unsupervised Learning Techniques, to fall asleep to

Chapter eight. Unsupervised learning techniques.

---

Yann LeCun, a Turing Award winner and Meta's chief AI scientist, said at NeurIPS in 2016 that if intelligence were a cake, unsupervised learning would be the cake itself, supervised learning the icing, and reinforcement learning the cherry on top. Unsupervised learning has huge, largely untapped potential, because most available data is unlabeled: the input features are there, but the labels are not.

A system that photographs items on a production line to detect defective ones can easily collect thousands of pictures a day, but none of them are labeled. Training a regular binary classifier requires human experts to label every picture as defective or normal, which is long, costly, and tedious, so usually only a small subset gets labeled and the classifier performs disappointingly. Every product change restarts the process. Unsupervised learning exploits the unlabeled data directly.

Dimensionality reduction, the most common unsupervised task, was covered in the previous chapter. This chapter covers three more.

Clustering groups similar instances together into clusters.

Anomaly detection, also called outlier detection, learns what normal data looks like and uses that to detect abnormal instances. The abnormal instances are called anomalies, or outliers, and the normal ones are called inliers. It is used for fraud detection, detecting defective products, identifying new trends in time series, and removing outliers from a dataset before training another model, which can significantly improve that model.

Density estimation is the task of estimating the probability density function, or PDF, of the random process that generated the dataset. It is commonly used for anomaly detection, since instances in very low-density regions are likely anomalies. It is also useful for data analysis and visualization.

---

Clustering. A hiker who finds several plants of an unknown kind can tell they probably belong to the same species without a botanist. Identifying groups of similar-looking objects needs no expert. Clustering is the task of identifying similar instances and assigning them to clusters, meaning groups of similar instances.

As in classification, each instance is assigned to a group. Unlike classification, clustering is unsupervised: there are no labels, so the algorithm must work out how to group the instances on its own.

The iris dataset shows the difference. With each instance's species known, it suits classifiers such as logistic regression or random forests. Without the labels, clustering algorithms take over. In a plot of two features, many of them easily detect the lower-left cluster, but it is not obvious that the upper-right cluster is made of two distinct subclusters. Using all four features, clustering algorithms identify the three clusters fairly well: a Gaussian mixture model assigns only five of the one hundred and fifty instances to the wrong cluster.

---

Applications of clustering.

Customer segmentation clusters customers by their purchases and website activity, to understand what they need and adapt products and marketing campaigns to each segment. Recommender systems can suggest content that other users in the same cluster enjoyed.

Data analysis. When exploring a new dataset, it can help to cluster it and analyze each cluster separately.

Dimensionality reduction. Affinity is any measure of how well an instance fits into a cluster. Each instance's feature vector can be replaced with the vector of its affinities to the k clusters. This is typically much lower-dimensional than the original vector, but can keep enough information for further processing.

Feature engineering. Cluster affinities are often useful as extra features. In chapter two, k-means added geographic cluster affinity features to the California housing dataset, which improved performance.

Anomaly detection. An instance with low affinity to all clusters is likely an anomaly, such as a website user making an unusual number of requests per second.

Semi-supervised learning. With only a few labels, clustering can propagate them to all the instances in the same cluster, greatly increasing the labels available to a supervised algorithm.

Search engines. To find images similar to a reference image, all images in the database are clustered, the reference image is assigned to a cluster by the trained model, and all images from that cluster are returned.

Image segmentation. Clustering pixels by color and replacing each pixel's color with the mean color of its cluster considerably reduces the number of colors in an image. Many object detection and tracking systems use it, because it makes each object's contour easier to detect.

There is no universal definition of a cluster. It depends on the context, and different algorithms capture different kinds of clusters. Some look for instances centered around a particular point, called a centroid. Others look for continuous regions of densely packed instances, which can take any shape. Some are hierarchical, looking for clusters of clusters.

---

K-means. K-means is a simple algorithm that clusters datasets made of blobs very quickly and efficiently, often in a few iterations. Stuart Lloyd proposed it at Bell Labs in 1957 as a technique for pulse-code modulation, but it was only published outside the company in 1982. Edward W. Forgy published virtually the same algorithm in 1965, so k-means is sometimes called the Lloyd–Forgy algorithm.

The book's example is an unlabeled dataset of five blobs. Scikit-Learn's k-means class tries to find each blob's center and assign each instance to the closest one. The number of clusters, k, must be specified. Here it is obviously five, but in general it is not that easy.

In clustering, an instance's label is the index of the cluster the algorithm assigns it to, not a target as in classification. The trained model stores its training instances' labels and the centroids it found as attributes. New instances are assigned to the cluster whose centroid is closest. The decision boundaries form a Voronoi tessellation, a partition of the plane into one region per centroid.

A few instances near the boundary between two clusters are probably mislabeled. K-means does not behave well when the blobs have very different diameters, because it only considers the distance to each centroid.

Assigning each instance to a single cluster is called hard clustering. Giving each instance a score per cluster is called soft clustering. The score can be the distance to the centroid, or a similarity score, also called affinity, such as the Gaussian radial basis function. The transform method of the k-means class returns the distance from each instance to every centroid. This turns a high-dimensional dataset into a k-dimensional one, which can be a very efficient nonlinear dimensionality reduction technique, and the distances can also serve as extra features for another model.

---

The k-means algorithm. Given the centroids, every instance could be labeled by assigning it to the cluster with the closest centroid. Given the labels, each centroid could be found by averaging the instances in its cluster. Neither is given at the start.

So the algorithm places the centroids randomly, for example by picking k instances at random and using their locations. It then labels the instances, updates the centroids, labels the instances again, updates the centroids again, and so on, until the centroids stop moving. It is guaranteed to converge in a finite number of steps, usually a small number, because the mean squared distance between the instances and their closest centroids can only go down at each step, and it cannot be negative. On the five-blob dataset, three iterations reach a clustering close to optimal.

The computational complexity is generally linear in the number of instances m, the number of clusters k, and the number of dimensions n, but only when the data has a clustering structure. Without one, in the worst case, the complexity can grow exponentially with the number of instances. In practice this rarely happens, and k-means is generally one of the fastest clustering algorithms.

Convergence is guaranteed, but not to the right solution. The algorithm may converge to a local optimum, depending on the centroid initialization. Unlucky random initializations can produce clearly suboptimal clusterings.

---

Centroid initialization methods. If the approximate centroid locations are known, for example from another clustering algorithm run earlier, the init hyperparameter can be set to an array of those centroids.

Another solution is to run the algorithm several times with different random initializations and keep the best solution. The n init hyperparameter controls the number of runs. With random initialization it defaults to ten, so calling the fit method runs the whole algorithm ten times.

The best solution is chosen with a performance metric called the model's inertia: the sum of the squared distances between each instance and its closest centroid. In the book's example, two suboptimal solutions have inertias of about two hundred and twenty and six hundred, while the good solution has about two hundred and twelve. The class keeps the model with the lowest inertia. Its score method returns the negative inertia, because Scikit-Learn's score methods follow the rule that greater is better.

K-means plus plus is an important improvement, proposed in 2006 by David Arthur and Sergei Vassilvitskii. Its smarter initialization tends to select centroids that are far from one another, making k-means much more likely to find all the important clusters and less likely to converge to a suboptimal solution, the way spreading out fishing boats helps find more schools of fish. The extra computation is well worth it, because it drastically reduces the number of runs needed.

The first centroid is an instance chosen uniformly at random. Each new centroid is an instance chosen at random with probability proportional to its squared distance to the closest centroid already chosen, so instances far from existing centroids are much more likely to be picked. This repeats until all k centroids are chosen.

K-means plus plus is the default initialization in Scikit-Learn, which actually uses a variant called greedy k-means plus plus: at each iteration it samples several candidate centroids and picks the best one. With this initialization, n init defaults to one.

---

Accelerated k-means and mini-batch k-means. In a 2003 paper, Charles Elkan showed that on some large datasets with many clusters, k-means can be accelerated by avoiding many unnecessary distance calculations. He used the triangle inequality, the fact that a straight line is always the shortest distance between two points, and kept track of lower and upper bounds on the distances between instances and centroids. Elkan's algorithm does not always accelerate training, and it can even slow training down significantly, depending on the dataset. Setting the algorithm hyperparameter to elkan enables it.

In a 2010 paper, David Sculley proposed mini-batch k-means, which uses a mini-batch instead of the full dataset at each iteration, moving the centroids just slightly each time. This speeds up the algorithm and makes it possible to cluster huge datasets that do not fit in memory. Scikit-Learn's mini-batch k-means class is used like the regular one. For data that does not fit in memory, the simplest option is NumPy's memory map class, as with incremental PCA. The partial fit method also works, but then multiple initializations and choosing the best must be handled by hand.

---

Finding the optimal number of clusters. Setting k to the wrong value can give quite bad results. On the five-blob dataset, k equal to three merges separate clusters, and k equal to eight chops some clusters into pieces.

Picking the model with the lowest inertia does not work. The inertia is about six hundred and fifty-three for k equal to three, about two hundred and twelve for k equal to five, and only about one hundred and twenty-seven for k equal to eight. Inertia keeps falling as k grows, because with more clusters each instance is closer to its nearest centroid.

Plotting inertia against k usually gives a curve shaped like an arm, with an inflection point called the elbow. Here the inertia drops very quickly up to k equal to four, then much more slowly, so four would look like a good choice: fewer would be dramatic, and more would just split good clusters in half. This method is rather coarse.

---

The silhouette score. It is a more precise, but more computationally expensive, way to choose k: the mean silhouette coefficient over all instances.

An instance's silhouette coefficient is b minus a, divided by the larger of a and b. Here a is the mean distance to the other instances in the same cluster, the mean intra-cluster distance. And b is the mean nearest-cluster distance: the mean distance to the instances of the next closest cluster, meaning the other cluster that minimizes b.

The coefficient ranges from minus one to plus one. Close to plus one means the instance is well inside its own cluster and far from the others. Close to zero means it is near a cluster boundary. Close to minus one means it may be in the wrong cluster. Scikit-Learn's silhouette score function takes the instances and their assigned labels. For the five-cluster model it gives about zero point six six.

Plotting the silhouette score against k is much more informative than the inertia plot. It confirms that k equal to four is a very good choice, but also shows that five is quite good too, and much better than six or seven.

A silhouette diagram is even more informative. It plots every instance's coefficient, sorted by cluster and by value, giving one knife shape per cluster. A shape's height is the number of instances in the cluster, and its width shows the sorted coefficients, where wider is better. A dashed line marks the mean silhouette score, and a cluster whose instances mostly fall short of it is rather bad, because they are too close to other clusters. For the five-blob data, three or six clusters give bad clusters, while four and five look good. With four, one cluster is rather big; with five, all clusters have similar sizes. So even though the overall score is slightly higher for four, five is the better choice.

---

Limits of k-means. K-means is fast and scalable, but it has to be run several times to avoid suboptimal solutions, and the number of clusters must be specified, which can be a hassle. It also does not behave well when clusters have varying sizes, different densities, or nonspherical shapes.

On a dataset of three ellipsoidal clusters with different dimensions, densities, and orientations, neither of two k-means solutions is good. The better one still chops off twenty-five percent of the middle cluster and assigns it to the cluster on the right. The other is terrible, even though its inertia is lower. Different clustering algorithms suit different data, and Gaussian mixture models work great on elliptical clusters.

The input features should be scaled before running k-means, or the clusters may be very stretched and k-means will perform poorly. Scaling does not guarantee nice spherical clusters, but it generally helps.

---

Clustering for image segmentation. Image segmentation partitions an image into multiple segments. In color segmentation, pixels with a similar color go in the same segment. This is enough for many applications, such as measuring the total forest area in a region from satellite images. In semantic segmentation, all pixels belonging to the same object type go in the same segment; in a self-driving car's vision system, all pedestrian pixels would form one pedestrian segment. In instance segmentation, all pixels belonging to the same individual object go in the same segment, so each pedestrian gets a separate segment.

The state of the art in semantic and instance segmentation uses complex architectures based on convolutional neural networks or vision transformers. Color segmentation is much simpler and can be done with k-means.

The book loads a picture of a ladybug with the Pillow package. The image is a three-dimensional array: height, width, and color channels. Here the channels are red, green, and blue, called RGB, each an unsigned eight-bit integer from zero to two hundred and fifty-five. Grayscale images have one channel, and some images have more, such as an alpha channel for transparency, or infrared channels in satellite images.

The array is reshaped into a long list of RGB colors, which k-means groups into eight clusters. Each pixel's color is replaced with the mean color of its cluster, and the result is reshaped back to the image's shape. With fewer than eight clusters, the ladybug's flashy red fails to get a cluster of its own and merges with colors from the surroundings, because k-means prefers clusters of similar sizes and the ladybug is small.

---

Clustering for semi-supervised learning. Semi-supervised learning applies when there are plenty of unlabeled instances and very few labeled ones. Clustering can help choose which instances to label, for example those near the cluster centroids. It can also propagate the most common label in each cluster to the unlabeled instances in that cluster.

The book uses the digits dataset, a simple MNIST-like dataset of one thousand seven hundred and ninety-seven grayscale eight by eight images of digits, with fourteen hundred for training and labels assumed for only fifty. A logistic regression model trained on fifty random labeled instances reaches seventy-five point eight percent test accuracy. Trained on the full labeled training set, it would reach about ninety point nine percent.

Clustering the training set into fifty clusters and taking the image closest to each centroid gives fifty representative images. Labeling those by hand and training on them raises accuracy to eighty-three point four percent, still with only fifty labels. Since labeling is costly, it is better to label representative instances than random ones.

Label propagation goes further: each representative's label is copied to all the other instances in its cluster. Training on the propagated training set raises accuracy to eighty-six point nine percent.

Dropping, in each cluster, the fifty percent of instances farthest from the center removes some outliers. Training on the rest gives eighty-eight point four percent, close to the fully labeled result, with only about five labels per class. This works partly because outliers were dropped, and partly because the propagated labels are very good: about ninety-eight point nine percent are correct.

Scikit-Learn's semi-supervised package has two classes, label spreading and label propagation, that build a similarity matrix between all instances and iteratively propagate labels to similar unlabeled instances. Its self-training classifier trains a base classifier, such as a random forest, on the labeled instances, adds its most confident predictions on unlabeled instances to the training set, and repeats until it cannot add more labels. These techniques are not magic bullets, but they occasionally give a model a small boost.

---

Active learning. To keep improving the model and the training set, the next step can be a few rounds of active learning, in which a human expert interacts with the learning algorithm and provides labels for specific instances when the algorithm requests them.

One of the most common strategies is uncertainty sampling. The model is trained on the labeled instances gathered so far and makes predictions on all the unlabeled instances. The instances it is most uncertain about, where its estimated probability is lowest, go to the expert for labeling. The process repeats until the performance improvement is no longer worth the labeling effort.

Other strategies label the instances that would cause the largest model change, or the largest drop in validation error, or the instances that different models, such as a support vector machine and a random forest, disagree on.

---

DBSCAN. Density-based spatial clustering of applications with noise, or DBSCAN, defines clusters as continuous regions of high density. Its approach, based on local density estimation, can identify clusters of any shape.

For each instance, the algorithm counts how many instances lie within a small distance, epsilon, of it. This region is the instance's epsilon neighborhood. If an instance has at least a minimum number of instances in its epsilon neighborhood, set by the min samples hyperparameter and counting the instance itself, it is a core instance. Core instances are those in dense regions. All instances in the neighborhood of a core instance belong to the same cluster. That neighborhood may contain other core instances, so a long chain of neighboring core instances forms a single cluster. Any instance that is not a core instance and has no core instance in its neighborhood is an anomaly.

DBSCAN works well when all clusters are well separated by low-density regions. On the moons dataset, with epsilon at zero point zero five and a minimum of five samples, Scikit-Learn's DBSCAN class labels many instances as anomalies, giving them a cluster index of minus one, and finds seven clusters. Raising epsilon to zero point two widens each neighborhood and gives a clustering that looks perfect.

The DBSCAN class has no predict method, only a fit predict method, so it cannot assign new instances to clusters. Its authors left the choice of classifier to the user, since different ones suit different tasks. The book trains a k-nearest neighbors classifier with fifty neighbors on the core instances and their cluster labels, which can then predict the cluster of new instances and a probability for each cluster. It could instead be trained on all instances, or all but the anomalies, depending on the task.

Because its training set has no anomalies, this classifier always picks a cluster, even a distant one. Adding a maximum distance fixes that: the classifier's k neighbors method returns the distance to the nearest core instance, and any new instance farther than zero point two is labeled an anomaly.

DBSCAN is simple yet powerful. It can identify any number of clusters of any shape, it is robust to outliers, and it has just two hyperparameters, epsilon and min samples. It can struggle when density varies significantly across clusters, or when some clusters have no sufficiently low-density region around them. Its computational complexity is roughly m squared times n, so it does not scale well to large datasets. Hierarchical DBSCAN, or HDBSCAN, also in Scikit-Learn, is often better at finding clusters of varying densities.

---

Other clustering algorithms in Scikit-Learn.

Agglomerative clustering builds a hierarchy of clusters from the bottom up, like tiny bubbles on water gradually attaching to each other. At each iteration it connects the nearest pair of clusters, starting from individual instances, which gives a binary tree of clusters whose leaves are the instances. It captures clusters of various shapes, gives a flexible cluster tree instead of forcing a choice of cluster scale, and works with any pairwise distance. It scales to large datasets only if given a connectivity matrix, a sparse m by m matrix indicating which pairs of instances are neighbors.

BIRCH, which stands for balanced iterative reducing and clustering using hierarchies, was designed for very large datasets. With fewer than about twenty features, it can be faster than batch k-means with similar results. It builds a tree holding just enough information to quickly assign new instances to clusters without storing them all, so it uses limited memory.

Mean-shift places a circle centered on each instance, then repeatedly shifts each circle to center it on the mean of the instances inside it, until the circles stop moving. The circles climb toward higher density until each reaches a local density maximum, and instances whose circles settle in the same place form a cluster. Like DBSCAN, it finds any number of clusters of any shape and relies on local density estimation, with a single hyperparameter, the circle radius, called the bandwidth. Unlike DBSCAN, it tends to chop clusters with internal density variations into pieces. Its complexity is m squared times n, so it does not suit large datasets.

Affinity propagation has instances exchange messages until each has elected another instance, or itself, to represent it. The elected instances are called exemplars, and each exemplar with its voters forms a cluster, much as voters choose a candidate who shares their opinions but is also popular. Exemplars tend to lie near cluster centers, as in k-means, but the number of clusters is determined during training, and clusters of different sizes are handled well. Its complexity is m squared, so it does not suit large datasets.

Spectral clustering builds a low-dimensional embedding from a similarity matrix between the instances, then runs another clustering algorithm in that space; Scikit-Learn's uses k-means. It can capture complex cluster structures and can cut graphs, for example to find clusters of friends on a social network. It does not scale well, and it behaves poorly when clusters have very different sizes.

---

Gaussian mixtures. A Gaussian mixture model, or GMM, is a probabilistic model that assumes the instances were generated from a mixture of several Gaussian distributions with unknown parameters. The instances generated from one Gaussian form a cluster that typically looks like an ellipsoid, and each cluster can have its own shape, size, density, and orientation. Which Gaussian generated each instance is unknown. K-means, by contrast, implicitly assumes clusters of similar size and density that are all roughly round.

In the simplest variant, implemented by Scikit-Learn's Gaussian mixture class, the number of Gaussian distributions, k, must be known in advance. The dataset is assumed to be generated like this. For each instance, a cluster is picked at random from the k clusters, and the probability of picking a given cluster is that cluster's weight, written with the Greek letter phi. The chosen cluster's index for each instance is called z. If an instance was assigned to a cluster, its location is sampled from a Gaussian distribution with that cluster's mean, mu, and covariance matrix, sigma.

Given a dataset, the first step is usually to estimate the weights and all the means and covariance matrices. The book fits the Gaussian mixture class with three components and ten initializations. The data was generated with two clusters of five hundred instances and one of two hundred and fifty, so the true weights are zero point four, zero point four, and zero point two. The estimated weights are about zero point four, zero point two one, and zero point three nine, in a different order, and the estimated means and covariance matrices are also close to the true ones.

---

Expectation maximization. The Gaussian mixture class relies on the expectation-maximization algorithm, or EM, which has much in common with k-means. It initializes the cluster parameters randomly, then repeats two steps until convergence: first assigning instances to clusters, which is the expectation step, then updating the clusters, which is the maximization step.

In clustering terms, EM generalizes k-means. It finds not only the cluster centers, the means, but also each cluster's size, shape, and orientation, given by the covariance matrices, and the clusters' relative weights. Unlike k-means, it uses soft assignments. In the expectation step, it estimates the probability that each instance belongs to each cluster, based on the current parameters. In the maximization step, each cluster is updated using all the instances, each weighted by its estimated probability of belonging to that cluster. These probabilities are called the clusters' responsibilities for the instances. Each cluster's update is driven mostly by the instances it is most responsible for.

Like k-means, EM can converge to poor solutions, so it must be run several times, keeping the best solution. That is why the book sets n init to ten; by default it is one. In the example it converges in four iterations.

The predict method assigns each instance to the most likely cluster, which is hard clustering, and the predict proba method gives the probability of belonging to each cluster, which is soft clustering.

A Gaussian mixture model is a generative model: new instances can be sampled from it, and they come out ordered by cluster index. Its score samples method estimates, at each given location, the log of the probability density function; the greater the score, the higher the density. The exponential of a score is the value of the PDF. These are probability densities, not probabilities, so they can take any positive value. The probability of falling in a region is the integral of the PDF over that region, and over the whole space it is one.

The model finds an excellent solution on the example data, but the task was easy: the data came from two-dimensional Gaussians, and the correct number of clusters was given. Real data is not always so Gaussian or low-dimensional.

---

Constraining the covariance matrices. With many dimensions, many clusters, or few instances, EM can struggle to converge to the optimal solution. The task can be made easier by limiting the number of parameters to learn, which means restricting the shapes and orientations the clusters can have. The covariance type hyperparameter does this.

Spherical means all clusters must be spherical, though with different diameters, meaning different variances. Diag means clusters can take any ellipsoidal shape and size, but the ellipsoid's axes must be parallel to the coordinate axes, so the covariance matrices are diagonal. Tied means all clusters share the same ellipsoidal shape, size, and orientation, meaning the same covariance matrix. With the default, full, each cluster can take any shape, size, and orientation, with its own unconstrained covariance matrix.

The cost of training depends on the number of instances m, the number of dimensions n, the number of clusters k, and the covariance constraints. With spherical or diag, it is on the order of k times m times n, assuming the data has a clustering structure. With tied or full, it is on the order of k times m times n squared, plus k times n cubed, so it does not scale to large numbers of features.

---

Gaussian mixtures for anomaly detection. Any instance in a low-density region can be considered an anomaly. The density threshold has to be chosen. A manufacturer trying to detect defective products usually knows the ratio of defective products, say two percent. The threshold is then set so that two percent of the instances lie in areas below it, which is the second percentile of the densities. Too many false positives, meaning good products flagged as defective, call for lowering the threshold. Too many false negatives, meaning defective products not flagged, call for raising it. This is the usual precision and recall trade-off.

Novelty detection is closely related. It differs from anomaly detection in that the algorithm is assumed to be trained on a clean dataset, not contaminated by outliers. Anomaly detection makes no such assumption, and in fact outlier detection is often used to clean up a dataset.

Gaussian mixture models try to fit all the data, outliers included. Too many outliers bias the model's view of normality, and some outliers may wrongly be considered normal. One fix is to fit the model once, use it to detect and remove the most extreme outliers, and fit it again on the cleaned dataset. Another is to use robust covariance estimation methods, such as Scikit-Learn's elliptic envelope class.

---

Selecting the number of clusters. Like k-means, the Gaussian mixture class needs the number of clusters. Inertia and the silhouette score are not reliable when clusters are not spherical or have different sizes, so they cannot be used here. Instead, the model can be chosen to minimize a theoretical information criterion.

The Bayesian information criterion, or BIC, equals the log of the number of instances times the number of parameters the model learns, minus twice the log of the maximized likelihood of the model. The Akaike information criterion, or AIC, equals twice the number of parameters minus twice the log of the maximized likelihood.

Both penalize models with more parameters to learn, such as more clusters, and reward models that fit the data well. They often select the same model. When they differ, the BIC's choice tends to be simpler, with fewer parameters, but fits the data a little less well, especially on larger datasets.

Scikit-Learn's Gaussian mixture class has methods that compute both. On the example data, both are lowest with three clusters, which is most likely the best choice.

---

The likelihood function. In everyday language, probability and likelihood are used interchangeably, but in statistics they differ. Given a statistical model with parameters theta, probability describes how plausible a future outcome x is when the parameter values are known. Likelihood describes how plausible a particular set of parameter values is after the outcome x is known.

The book's example is a one-dimensional mixture of two Gaussian distributions centered at minus four and plus one, with a single parameter, theta, controlling the standard deviation of both. The model is a function of both x and theta. Fixing theta at one point three gives a probability density function of x. The probability that x falls between minus two and plus two is the integral of that PDF over that range, the area under the curve. If theta is unknown and instead a single instance x equal to two point five has been observed, fixing x gives the likelihood function, a function of theta.

So the PDF is a function of x with theta fixed, and the likelihood function is a function of theta with x fixed. The likelihood function is not a probability distribution. Integrating a probability distribution over all values of x always gives one, but integrating the likelihood over all values of theta can give any positive value.

Estimating the most likely parameter values for a dataset means finding the values that maximize the likelihood function given the data. With the single observation of two point five, the maximum likelihood estimate, or MLE, of theta is about one point six six. If a prior probability distribution over theta exists, it can be taken into account by maximizing the likelihood times the prior, rather than the likelihood alone. This is called maximum a-posteriori, or MAP, estimation. Since MAP constrains the parameter values, it can be thought of as a regularized version of MLE.

Maximizing the likelihood is equivalent to maximizing its logarithm, because the logarithm is strictly increasing, and the log likelihood is generally easier to work with. With several independent observations, maximizing the product of their likelihoods is equivalent to maximizing the sum of their log likelihoods, which is much simpler, because the log of a product is the sum of the logs.

The likelihood evaluated at the best parameter value, for the dataset, is the maximized likelihood used in the AIC and BIC, a measure of how well the model fits the data.

---

Bayesian Gaussian mixture models. Instead of searching manually for the optimal number of clusters, Scikit-Learn's Bayesian Gaussian mixture class can give weights equal or close to zero to unnecessary clusters. The number of components is set to a value believed to be larger than the optimal number of clusters, which assumes some minimal knowledge of the problem, and the algorithm eliminates the unnecessary clusters automatically. With ten components on the example data, it gives weights of about zero point four, zero point two one, and zero point three nine to three clusters and zero to the other seven. The resulting clusters are almost identical to the earlier ones.

Gaussian mixture models work great on ellipsoidal clusters, but not on clusters of very different shapes. A Bayesian Gaussian mixture fit to the moons dataset searches for ellipsoids and finds eight clusters instead of two. Its density estimate is not too bad, so it could perhaps be used for anomaly detection, but it fails to identify the two moons.

---

Other algorithms for anomaly and novelty detection in Scikit-Learn.

Fast-MCD, short for minimum covariance determinant, is implemented by the elliptic envelope class and is useful for outlier detection, in particular to clean up a dataset. It assumes the inliers come from a single Gaussian distribution, not a mixture, and that the dataset is contaminated with outliers that did not. When it estimates the shape of the elliptic envelope around the inliers, it ignores the instances most likely to be outliers, which gives a better envelope and better outlier detection.

Isolation forest is an efficient outlier detection algorithm, especially for high-dimensional datasets. It builds a random forest in which each decision tree is grown randomly: at each node it picks a random feature, then a random threshold between that feature's minimum and maximum values, and splits the dataset in two. The dataset is gradually chopped into pieces until every instance is isolated from the others. Anomalies are usually far from other instances, so on average across the trees they get isolated in fewer steps than normal instances.

Local outlier factor, or LOF, is also good for outlier detection. It compares the density of instances around a given instance with the density around its neighbors. An anomaly is often more isolated than its k nearest neighbors.

One-class SVM is better suited to novelty detection. A kernelized support vector machine implicitly maps instances to a high-dimensional space and separates two classes there with a linear classifier. With only one class, the one-class SVM instead separates the instances from the origin in that space. In the original space, this amounts to finding a small region that encompasses all the instances, and a new instance outside it is an anomaly. Besides the usual kernelized SVM hyperparameters, a margin hyperparameter corresponds to the probability of a new normal instance being mistakenly considered novel. It works great, especially on high-dimensional data, but like all SVMs it does not scale to large datasets.

PCA, and other dimensionality reduction techniques with an inverse transform method, can detect anomalies too. The reconstruction error of an anomaly is usually much larger than that of a normal instance. This is a simple and often quite efficient approach.

---

Summary.

Most available data is unlabeled. Unsupervised learning includes dimensionality reduction, clustering, anomaly detection, and density estimation.

Clustering groups similar instances, with clusters defined by centroids, by dense regions, or as a hierarchy. K-means alternates between assigning instances to the nearest centroid and moving each centroid to the mean of its instances. It is fast and scalable, but needs the number of clusters, can converge to local optima, and struggles with clusters of different sizes, densities, or nonspherical shapes. K-means plus plus and multiple runs keeping the lowest inertia reduce bad solutions, mini-batch k-means handles huge datasets, and the number of clusters can be chosen with the inertia elbow or, more precisely, with silhouette scores.

Clustering can segment images by color and power semi-supervised learning: labeling representative instances and propagating their labels can approach fully supervised accuracy with very few labels. Active learning asks an expert to label the instances the model is least sure about.

DBSCAN defines clusters as continuous dense regions, finds clusters of any shape, and flags isolated instances as anomalies.

A Gaussian mixture model assumes the data comes from several Gaussian distributions and fits them with expectation maximization, a soft-assignment generalization of k-means. It supports clustering, sampling, density estimation, and anomaly detection by density threshold. The number of clusters is chosen by minimizing the BIC or AIC, or automatically with a Bayesian Gaussian mixture. It suits ellipsoidal clusters, not arbitrary shapes.

Other anomaly and novelty detectors include Fast-MCD, isolation forest, local outlier factor, one-class SVM, and PCA reconstruction error.
