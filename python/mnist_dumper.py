"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip
import json, codecs
from json import JSONEncoder

# Third-party libraries
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if type(np.array(o)).__module__ == np.__name__:
            return o.tolist()
        else:
            return o


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def dump_data():
    tr_d, va_d, te_d = load_data()
    print("dumping training")

    with open('../target/mnist_training.json', 'w') as f:
        f.write(NumpyArrayEncoder().encode(tr_d))

    print("dumping validation")
    with open('../target/mnist_validation.json', 'w') as f:
        f.write(NumpyArrayEncoder().encode(va_d))

    print("dumping testing")
    with open('../target/mnist_testing.json', 'w') as f:
        f.write(NumpyArrayEncoder().encode(te_d))


dump_data()
