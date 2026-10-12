# Chapter 1 — The Machine Learning Landscape, to fall asleep to

Chapter one. The machine learning landscape.

---

Machine learning is the science of programming computers so they can learn from data.

Arthur Samuel defined it in 1959 as the field of study that gives computers the ability to learn without being explicitly programmed. Tom Mitchell's 1997 definition is more precise: a program learns from experience E, with respect to a task T and a performance measure P, if its performance on T, as measured by P, improves with experience E.

A spam filter is a machine learning program. Its task is to flag spam in new emails. Its experience is the training data: example emails flagged as spam by users, and examples of regular email, called ham. Its performance measure could be the ratio of correctly classified emails, which is called accuracy. Accuracy is a common performance measure for classification tasks.

The examples a system learns from form the training set. Each example is a training instance, also called a sample. The part of the system that learns and makes predictions is the model. Neural networks and random forests are examples of models.

Downloading a copy of Wikipedia gives a computer more data, but it does not make it better at any task. That is not machine learning.

---

A spam filter written with traditional programming is a long list of hand-written rules, one per pattern noticed in spam, such as certain words in the subject line. It is hard to maintain, and every time spammers change tactics, someone has to write new rules.

A machine learning spam filter learns which words and phrases predict spam by finding patterns that are unusually frequent in spam compared with ham. The program is shorter, easier to maintain, and usually more accurate. When spammers change tactics, for example writing "for you" instead of the number four and the letter U, the filter notices that the new phrase has become frequent in flagged spam and adapts on its own.

Machine learning also handles problems that are too complex for traditional approaches, or that have no known algorithm. Speech recognition is an example: hand-coded rules cannot cover thousands of words spoken by millions of people in noisy environments and many languages. The best approach is an algorithm that learns from many recorded examples of each word.

Machine learning can also help humans learn. A trained model can be inspected to see what it learned, which can reveal unsuspected correlations and trends. Digging into large amounts of data to discover hidden patterns is called data mining.

Machine learning is useful for four kinds of problems. First, problems whose existing solutions need long lists of hand-tuned rules. Second, complex problems where traditional approaches yield no good solution. Third, fluctuating environments, because a machine learning system can be retrained on new data. Fourth, getting insights about complex problems and large amounts of data.

---

Examples of machine learning applications, and the techniques used for each.

Classifying images of products on a production line is image classification, done with convolutional neural networks or vision transformers.

Detecting tumors in brain scans is semantic image segmentation: every pixel is classified, to find the exact location and shape of the tumor.

Classifying news articles, flagging offensive comments, and summarizing long documents are natural language processing tasks: text classification and text summarization. Transformers work best for these.

Estimating genetic disease risk from a very long DNA sequence requires finding patterns spread across long sequences, which is where state space models do well.

Creating a chatbot or personal assistant involves natural language understanding and question answering.

Forecasting next year's revenue from performance metrics is a regression task, meaning predicting a value. Any regression model can be used, such as linear regression, polynomial regression, a random forest, or a neural network. To account for sequences of past metrics, recurrent networks, convolutional networks, or transformers can be used.

Making an app respond to voice commands is speech recognition. Audio samples are long, complex sequences, typically processed with recurrent networks, convolutional networks, or transformers.

Detecting credit card fraud is anomaly detection, done with isolation forests, Gaussian mixture models, or autoencoders.

Segmenting clients by their purchases is clustering, done with k-means, DBSCAN, and other algorithms.

Representing a complex, high-dimensional dataset in a clear diagram is data visualization, often using dimensionality reduction.

Recommending products based on past purchases is a recommender system. One approach is a neural network trained on past sequences of purchases across all clients, which outputs the most likely next purchase.

Building an intelligent bot for a game is often done with reinforcement learning, which trains an agent to choose the actions that maximize its rewards over time within an environment. AlphaGo, which beat the world champion at Go, was built with reinforcement learning.

---

Machine learning systems are classified by three criteria. First, how they are supervised during training: supervised, unsupervised, semi-supervised, self-supervised, or reinforcement learning. Second, whether they can learn incrementally on the fly: online learning versus batch learning. Third, whether they compare new data points to known ones, or detect patterns in the training data and build a predictive model: instance-based learning versus model-based learning.

The criteria are not exclusive. A spam filter that learns on the fly, using a deep neural network trained on examples of spam and ham, is an online, model-based, supervised learning system.

---

Supervised learning. The training data includes the desired solutions, called labels.

Classification is a typical supervised task. A spam filter is trained on emails labeled spam or ham, and learns to classify new emails.

Regression is the other typical supervised task: predicting a numeric value, called the target, from a set of features. For example, predicting a car's price from its mileage, age, and brand. Training requires many examples of cars with both their features and their prices.

Some regression models can be used for classification, and vice versa. Logistic regression is commonly used for classification, because it outputs the probability of belonging to a class, for example a twenty percent chance of being spam.

Target and label are generally synonyms. Target is more common in regression, label in classification. Features are also called predictors or attributes.

---

Unsupervised learning. The training data is unlabeled. The system learns without a teacher.

Clustering detects groups of similar instances. Run on data about a blog's visitors, a clustering algorithm might find that forty percent are teenagers who like comic books and read after school, and twenty percent are adults who like science fiction and visit on weekends, without being told the groups exist. A hierarchical clustering algorithm also subdivides each group into smaller groups.

Visualization algorithms take complex, unlabeled data and output a two- or three-dimensional representation that can be plotted. They try to preserve as much structure as possible, for example keeping separate clusters from overlapping.

Dimensionality reduction simplifies the data without losing too much information. One way is to merge correlated features. A car's mileage is strongly correlated with its age, so the two can be merged into one feature representing wear and tear. This is called feature extraction. Reducing dimensions before feeding data to another algorithm makes it run faster, uses less disk and memory, and can improve performance.

Anomaly detection learns from mostly normal instances, then flags new instances that do not look normal. Examples are unusual credit card transactions, manufacturing defects, and outliers to remove from a dataset before training another algorithm.

Novelty detection is similar, but it detects new instances that differ from everything in the training set, and it requires a clean training set containing none of the instances to detect. If one percent of a dog photo dataset is Chihuahuas, a novelty detector should not treat new Chihuahua photos as novelties, while an anomaly detector may treat them as anomalies because they are rare and different.

Association rule learning finds interesting relations between attributes in large amounts of data. Sales logs might show that people who buy barbecue sauce and potato chips also tend to buy steak.

---

Semi-supervised learning. Labeling data is time-consuming and costly, so datasets often have many unlabeled instances and few labeled ones. Semi-supervised algorithms work with partially labeled data.

A photo-hosting service is an example. It clusters photos by the people in them, which is the unsupervised part. The user then labels each person once, and the service can name everyone in every photo.

Most semi-supervised algorithms combine unsupervised and supervised algorithms. For example, a clustering algorithm groups similar instances, each unlabeled instance gets the most common label in its cluster, and then any supervised algorithm can train on the fully labeled dataset.

---

Self-supervised learning generates a fully labeled dataset from a fully unlabeled one.

For example, a random part of each image in a large unlabeled dataset is masked, and a model is trained to recover the original image. The masked images are the inputs and the original images are the labels.

The resulting model can repair damaged images or remove unwanted objects, but usually the goal is to fine-tune it for a different task. A model that can repair masked photos of pets must already distinguish pet species, since it must not put a dog's face on a cat. It can be adapted to predict pet species, then fine-tuned on a small labeled dataset to map the species it already knows to the expected labels.

Transferring knowledge from one task to another is called transfer learning. It is one of the most important techniques in machine learning, especially with deep neural networks.

Large language models are trained in a similar way, by masking words in a huge text corpus and predicting the missing words. The pretrained model is then fine-tuned for applications from sentiment analysis to chatbots.

Self-supervised learning uses unlabeled data, but it trains with generated labels and focuses on the same tasks as supervised learning, mainly classification and regression. Unsupervised learning usually refers to clustering, dimensionality reduction, and anomaly detection. Self-supervised learning is best treated as its own category.

---

Reinforcement learning. The learning system, called an agent, observes an environment, selects and performs actions, and gets rewards, or penalties in the form of negative rewards. It must learn the best strategy, called a policy, to get the most reward over time. A policy defines which action to choose in a given situation.

Many robots use reinforcement learning to learn to walk. DeepMind's AlphaGo learned its policy by analyzing millions of games and then playing against itself, and in 2017 it beat Ke Jie, the world's top-ranked Go player. Learning was turned off during those games; AlphaGo only applied the policy it had learned.

---

Batch learning versus online learning.

In batch learning, the system is trained on all the available data at once. This takes a lot of time and computing resources, so it is done offline: the system is trained, launched into production, and then runs without learning. This is called offline learning.

A model's performance decays over time because the world keeps changing while the model does not. This is called data drift, or model rot. The fix is to retrain regularly on up-to-date data. A model classifying cats and dogs decays slowly, though cameras, image formats, and brightness still change. A model making financial predictions decays fast.

To learn about new data, a batch system must be retrained from scratch on the full dataset, old and new data together, and then replace the old model. The process can be automated, but training on the full data can take hours, so it is usually done daily or weekly. It also costs a lot of computing resources, and it may be impossible with huge datasets or on devices with limited resources, such as a smartphone or a Mars rover.

In online learning, the system is trained incrementally by feeding it instances sequentially, one at a time or in small groups called mini-batches. Each learning step is fast and cheap, so the system learns about new data as it arrives. The most common online algorithm is gradient descent. Online learning suits systems that must adapt quickly, such as detecting new patterns in the stock market, and systems with limited computing resources.

Online learning algorithms can also train on datasets too large to fit in one machine's memory. The algorithm loads part of the data, runs a training step on it, and repeats until it has gone through all the data. This is called out-of-core learning. It is usually done offline, so online learning is best thought of as incremental learning.

The learning rate controls how fast an online system adapts to new data. A high learning rate adapts quickly but tends to forget old data, which is called catastrophic forgetting. A low learning rate gives the system more inertia: it learns more slowly, but is less sensitive to noise in new data and to sequences of nonrepresentative data points.

If bad data is fed to an online system, its performance declines, possibly quickly. Bad data can come from a bug, such as a malfunctioning sensor, or from someone trying to game the system, such as spamming a search engine. The system must be monitored closely, learning switched off and possibly reverted if performance drops, and input data monitored for anomalies.

---

Instance-based learning versus model-based learning. This criterion is about how a system generalizes, meaning how it makes good predictions on examples it has never seen. Good performance on the training data is not enough. The goal is good performance on new instances.

Instance-based learning learns the examples by heart, then generalizes to new cases by comparing them to the learned examples with a similarity measure. A spam filter could flag emails that share many words with known spam emails. Instance-based learning often works well with small datasets, especially when the data keeps changing. It scales poorly: it requires deploying a copy of the training set, searching for similar instances is slow, and it does not work well with high-dimensional data such as images.

Model-based learning builds a model of the examples and uses the model to make predictions.

The chapter's example asks whether money makes people happy, using life satisfaction scores from the OECD's Better Life Index and gross domestic product per person from the World Bank. Life satisfaction appears to rise roughly linearly with wealth, with noise. So life satisfaction is modeled as a linear function of wealth per person: theta zero plus theta one times wealth per person. Choosing this type of model is called model selection. Theta zero and theta one are the model's parameters. By adjusting them, the model can represent any linear function.

To choose parameter values, a performance measure is needed. A utility function, or fitness function, measures how good the model is. A cost function, or loss function, measures how bad it is. Linear regression uses a cost function that measures the distance between the model's predictions and the training examples, and the goal is to minimize it.

The linear regression algorithm takes the training examples and finds the parameter values that make the model fit the data best. This is called training the model. Applying the trained model to new cases is called inference. For a country whose wealth per person is about thirty-three thousand dollars, the model predicts a life satisfaction of about six.

The word model can mean a type of model, such as linear regression, a fully specified architecture, such as linear regression with one input and one output, or the final trained model with its parameter values. Model selection means choosing the type of model and its architecture. Training means running an algorithm to find the parameters that best fit the training data.

An instance-based alternative is k-nearest neighbors regression: find the k countries with the most similar wealth, and average their life satisfaction. With k equal to three, the prediction for the same country is about five point seven.

If predictions are poor, the options are to use more attributes, such as employment rate, health, and air pollution, to get more or better training data, or to select a more powerful model, such as polynomial regression.

A typical machine learning project follows four steps. Study the data. Select a model. Train it on the training data, meaning the learning algorithm searches for the parameter values that minimize a cost function. Apply the model to make predictions on new cases.

There are more categories. Ensemble learning trains several models and combines their predictions. Federated learning trains models across many devices, such as smartphones, and adapts them to each user without exchanging raw data, which protects privacy. Meta-learning trains models to learn new tasks quickly from little data.

---

The main challenges of machine learning come from two sources: bad data and bad models.

Insufficient quantity of training data. A child can learn what an apple is from a few examples. Most machine learning algorithms need thousands of examples for simple problems, and millions for complex problems such as image or speech recognition, unless parts of an existing model can be reused through transfer learning.

In 2001, Michele Banko and Eric Brill showed that very different algorithms, including fairly simple ones, performed almost identically on a complex natural language problem once they had enough data. In 2009, Peter Norvig and colleagues popularized this idea in a paper titled "The Unreasonable Effectiveness of Data". Small and medium datasets are still common, and extra data is not always cheap, so algorithms still matter.

Nonrepresentative training data. To generalize well, the training data must be representative of the new cases. The countries in the life satisfaction example left out very poor and very rich countries. With those countries added, the best linear fit changes significantly, and it becomes clear that a simple linear model will never work well: very rich countries are not happier than moderately rich ones, and some poor countries are happier than many rich ones.

A sample that is too small has sampling noise: nonrepresentative data as a result of chance. Even a very large sample can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

The best-known example of sampling bias is the 1936 United States presidential election poll by the Literary Digest. It mailed about ten million people, got two point four million answers, and predicted that Landon would win fifty-seven percent of the vote. Roosevelt won with sixty-two percent. The addresses came from telephone directories, magazine subscriber lists, and club membership lists, which favored wealthier people, who were more likely to vote Republican. And fewer than a quarter of the people polled answered, which is a special type of sampling bias called nonresponse bias.

Another example: building a training set of funk music videos by searching for "funk music" on YouTube assumes the search results are representative. In reality they are biased toward popular artists, and in Brazil they return a lot of "funk carioca", which sounds nothing like James Brown.

Poor-quality data. Errors, outliers, and noise make patterns harder to detect. Cleaning training data is often worth the effort, and data scientists spend a significant part of their time on it. Clear outliers can be discarded or fixed manually. For instances missing some features, for example customers who did not give their age, the options are to ignore the attribute, ignore those instances, fill in the missing values, for example with the median age, or train one model with the feature and one without.

Irrelevant features. Garbage in, garbage out. A system can only learn if the training data has enough relevant features and not too many irrelevant ones. Finding a good set of features is called feature engineering. It involves feature selection, which means choosing the most useful existing features; feature extraction, which means combining existing features into a more useful one, for example with dimensionality reduction; and creating new features by gathering new data.

---

Overfitting the training data. Overfitting means the model performs well on the training data but does not generalize well. It is like concluding that every taxi driver in a country is a thief after one driver overcharges you.

A high-degree polynomial model of life satisfaction fits the training data much better than a linear model, but its predictions cannot be trusted. Complex models such as deep neural networks can detect subtle patterns, but if the training set is noisy or too small, they detect patterns in the noise itself. Given country names as a feature, a model might find that every country in the training data with a W in its name, such as New Zealand, Norway, Sweden, and Switzerland, has a life satisfaction above seven. That pattern is due to chance, and it will not generalize to Rwanda or Zimbabwe. The model cannot tell a real pattern from noise.

Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. The solutions are to simplify the model, by choosing one with fewer parameters, reducing the number of attributes, or constraining the model; to gather more training data; and to reduce the noise in the training data, by fixing errors and removing outliers.

Constraining a model to make it simpler and reduce the risk of overfitting is called regularization. The linear model has two parameters, so it has two degrees of freedom: it can adjust the height of the line, theta zero, and its slope, theta one. Forcing theta one to zero leaves one degree of freedom, so the line can only move up and down, ending up near the mean. Allowing theta one to change but keeping it small gives somewhere between one and two degrees of freedom. The goal is a balance between fitting the training data well and keeping the model simple enough to generalize. A regularized model fits its training data less closely than an unregularized one, but can generalize better to new examples.

The amount of regularization is controlled by a hyperparameter. A hyperparameter is a parameter of the learning algorithm, not of the model. It is not affected by the learning algorithm: it is set before training and stays constant during training. A very large regularization hyperparameter gives an almost flat model, which will not overfit but is unlikely to find a good solution. Tuning hyperparameters is an important part of building a machine learning system.

Underfitting the training data is the opposite of overfitting. It happens when the model is too simple to learn the underlying structure of the data. A linear model of life satisfaction is prone to underfit, because reality is more complex than the model, so its predictions are inaccurate even on the training examples. The fixes are to select a more powerful model with more parameters, to feed the algorithm better features through feature engineering, and to reduce the constraints on the model, for example by reducing the regularization hyperparameter.

Deployment issues. A model that neither overfits nor underfits can still be too complex to maintain, too large to fit in memory, too slow, unable to scale, insecure, or outdated if not updated often. These operational problems need different skills from data modeling, so companies often have a dedicated machine learning operations team, called MLOps.

---

Testing and validating. The only way to know how well a model generalizes is to try it on new cases. Deploying it and monitoring it works, but users will notice if the model is bad.

The better option is to split the data into a training set and a test set. The model is trained on the training set and tested on the test set. The error rate on new cases is the generalization error, also called the out-of-sample error, and evaluating on the test set estimates it. A low training error with a high generalization error means the model is overfitting.

It is common to use eighty percent of the data for training and twenty percent for testing. With ten million instances, holding out one percent gives a test set of one hundred thousand instances, which is probably enough.

Hyperparameter tuning and model selection. Suppose many models with different regularization values are compared on the test set, and the best one has five percent error, but in production it has fifteen percent error. The generalization error was measured many times on the same test set, and the model and its hyperparameters were adapted to that particular set, so the estimate is too optimistic.

The common solution is holdout validation. Part of the training set is held out as a validation set, also called a development set, or dev set. Candidate models with various hyperparameters are trained on the reduced training set, and the one that performs best on the validation set is selected. The best model is then retrained on the full training set, including the validation set, and the final model is evaluated once on the test set to estimate the generalization error.

If the validation set is too small, evaluations are imprecise, and a suboptimal model may be selected. If it is too large, the candidates are trained on much less data than the final model, which is like choosing the fastest sprinter to run a marathon. Repeated cross-validation solves this by using many small validation sets: each model is evaluated once per validation set after training on the rest of the data, and the evaluations are averaged. The cost is that training time is multiplied by the number of validation sets.

---

Data mismatch. Sometimes plenty of training data is available, but it is not perfectly representative of production data. For example, an app that identifies flower species from phone photos can train on millions of flower pictures from the web, but those differ from photos taken with the app, of which there may be only a thousand.

The validation set and the test set must be as representative as possible of production data, so they are built only from representative pictures: half in the validation set and half in the test set, with no duplicates or near-duplicates across the two.

If performance on the validation set is then poor, the cause could be overfitting or the mismatch between web pictures and app pictures. To tell them apart, some of the web training pictures are held out in another set, which Andrew Ng called the train-dev set. The model is trained on the training set and evaluated on the train-dev set. If it performs poorly there, it has overfit the training set, so the fix is to simplify or regularize the model, get more training data, or clean the training data. If it performs well on the train-dev set but poorly on the dev set, the problem is data mismatch. One fix is to preprocess the web images to make them look more like the app's pictures, then retrain. Once the model performs well on both the train-dev set and the dev set, it is evaluated one final time on the test set.

---

The No Free Lunch theorem. A model is a simplified representation of the data, and choosing a type of model means making assumptions about the data. A linear model assumes the data is fundamentally linear and that the distance between the instances and the line is noise that can be ignored.

In 1996, David Wolpert showed that if no assumptions are made about the data, there is no reason to prefer one model over any other. This is the No Free Lunch theorem. For some datasets the best model is linear, for others it is a neural network, and no model is guaranteed in advance to work better. The only way to know which model is best is to evaluate them all, which is impossible, so in practice you make reasonable assumptions about the data and evaluate a few reasonable models: for example, linear models with various levels of regularization for simple tasks, and various neural networks for complex ones.

---

Summary.

Machine learning makes machines get better at a task by learning from data instead of following explicitly coded rules.

Machine learning systems can be supervised, unsupervised, semi-supervised, self-supervised, or reinforcement learning; batch or online; and instance-based or model-based.

A training set is fed to a learning algorithm. A model-based algorithm tunes parameters to fit the model to the training set, so it can make good predictions on new cases. An instance-based algorithm learns the examples by heart and compares new instances to them with a similarity measure.

A system performs poorly if the training set is too small, nonrepresentative, noisy, or full of irrelevant features. The model must be neither too simple, which causes underfitting, nor too complex, which causes overfitting. Deployment constraints matter too.

A model is evaluated on a test set it has never seen. Models and hyperparameters are compared on a validation set, or with cross-validation, and the test set is used once, at the end.
