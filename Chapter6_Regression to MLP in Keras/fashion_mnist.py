import urllib.request
import gzip
import idx2numpy


def load_data():
    """Loads the Fashion MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    path_x_train = urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
    path_y_train = urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    path_x_test = urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    path_y_test = urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    print(path_x_train)

    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        x_train = idx2numpy.convert_from_string(f.read())
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        y_train = idx2numpy.convert_from_string(f.read())
    with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
        x_test = idx2numpy.convert_from_string(f.read())
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
        y_test  = idx2numpy.convert_from_string(f.read())

    return (x_train, y_train), (x_test, y_test)
