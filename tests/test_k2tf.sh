#!/bin/bash
# It might seem stupid to test the k2tf module in a bash script, but as it
# turns out, the tensorflow graph manipulations that are done by the import, do
# not work in tensorflow's test_session.

set -xe

python -c '
import numpy as np

from keras import backend as K
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers.core import Dense

from ionn import k2tf


def model():
    network = Sequential()
    network.add(Dense(4,
                      init=Constant(value=0.1),
                      activation="relu",
                      name="fc",
                      input_shape=(2,),
                      )
                )
    return network

M = model()
k2tf.save_keras_model(M, "keras_model.pb", freeze=True)
'

python -c '
import tensorflow as tf
import numpy as np

from ionn import k2tf

specs = k2tf.load_keras_model("keras_model.pb")

with tf.Session() as sess:
    input_shape = tuple(1 if n is None else int(n) for n in specs.input_dims[0])
    output = sess.run(specs.outputs, feed_dict={specs.inputs[0]: np.ones(input_shape, "d")})

assert len(output) == 1, "output should contain a single tensor but contained {}".format(len(output))
assert np.array(output[0]).shape == (1, 4), "output tensor should have shape (1, 4)  but had {}".format(output.shape)
assert np.all(output[0] == 0.2), "output should be all 0.2 but is {}".format(output)
'

TEST_STATUS=$?

# Clean up
rm keras_model.pb
rm keras_model.pb.io

exit $TEST_STATUS
