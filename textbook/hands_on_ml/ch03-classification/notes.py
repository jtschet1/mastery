# download mnist dataset

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist.data, mnist.target

def plot_image(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

digit = X[0]
plot_image(digit)
plt.show()