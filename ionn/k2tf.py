#!/usr/bin/env python

import re
import json

from collections import namedtuple
import keras.backend.tensorflow_backend as K

from ionn import tfpb

ModelInfo = namedtuple('ModelInfo',
                       ['outputs', 'inputs', 'input_dims', 'nodes'])


# def compare_predictions():
#     keras_model = example_k.model()
#     tf_model = example_tf.model()
#
#     x = np.ones((1, 2), 'd')
#     pred_k = keras_model.predict(x)
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         pred_tf = sess.run(tf_model[1], feed_dict={tf_model[0]: x})
#     print('keras', pred_k)
#     print('tensorflow', pred_tf)
#     print('difference', pred_k - pred_tf)


def find_io_nodes(graphdef):
    io_nodes = {'input': [], 'output': []}

    inputs = get_nodes_that_are_further_used(graphdef)
    inputs = prune_potential_output_nodes(inputs)

    for node in graphdef.node:
        if node.op == 'Placeholder':
            io_nodes['input'].append(node.name)
        if node.name not in inputs:
            io_nodes['output'].append(node.name)
    io_nodes['output'] = prune_potential_output_nodes(io_nodes['output'])

    return io_nodes


def get_nodes_that_are_further_used(graphdef):
    inputs = []
    for node in graphdef.node:
        inputs.extend(node.input)
    return set(inputs)


def prune_potential_output_nodes(inputs, irrelevant=[r'.*/Assign']):
    inputs = set(inputs)
    for expr in irrelevant:
        matches = set([inp for inp in inputs if re.match(expr, inp)])
        inputs -= matches
    return list(inputs)


def save_io(model, fname):
    """Save meta-information that simplifies loading the model's protobuf

    Args:
        model:
            keras model object
        fname:
            name of the output file
    """
    input_nodes = [node.name for node in model.inputs]
    output_nodes = [node.name for node in model.outputs]
    input_dims = [node.get_shape().as_list() for node in model.inputs]
    with open(fname + '.io', 'w') as f:
        f.write(json.dumps({'input': input_nodes,
                            'output': output_nodes,
                            'input_dims': input_dims}))


def save_keras_model(model, fname='keras_model.pb', freeze=True):
    """Save keras model as tensorflow protobuf file

    Args:
        model:
            keras model object. Currently only sequential models with a single
            output tensor are supported.
        fname:
            name of the output protobuf file. The function will create a second
            file that contains metadata needed for loading (e.g. names of output
            nodes).
        freeze:
            boolean to indicate if the model should be "frozen". Note that
            freezing means that you won't be able to further train the model and
            you can only do prediction on a frozen model. Yet, not freezing the
            model may loose all information about variable parameters (i.e.
            trained weights), because those would be stored in checkpoint files
            for tensorflow that are not created by keras. Loading unfrozen
            models also don't really seem to work yet.
    """
    if not len(model.outputs) == 1:
        raise ValueError(
            'Can only store models with a single output tensor, found {}'
            .format(len(model.outputs)))

    graphdef = model.outputs[0].graph.as_graph_def()
    output_nodes = [node.name for node in model.outputs]

    tfpb.save_protobuf(
        graphdef, fname,
        output_nodes=output_nodes,
        freeze=freeze,
        session=K.get_session())

    save_io(model, fname)


def load_keras_model(fname, session=None):
    """Load a protobuf that was stored from a keras model into tensorflow

    Args:
        fname:
            name of the protobuf file of the model. The function will look for a
            second file with extension .io, which stores metadata needed to load
            the protobuf.

    Returns:
        a dictionary with the following keys:
            - input: are the names of the input nodes expected by the model
            - input_dims: are the dimensions of the input tensors
            - output: are the names of the output nodes returned by the model
            - nodes: are the tensorflow nodes that make up the model. These will
                already be merged with the current session
    """

    with open('{}.io'.format(fname), 'r') as f:
        io_nodes = json.loads(f.read())
    nodes = tfpb.load_protobuf(fname,
                               output_nodes=io_nodes['output'],
                               session=session)
    return ModelInfo(outputs=io_nodes['output'],
                     inputs=io_nodes['input'],
                     input_dims=io_nodes['input_dims'],
                     nodes=nodes)


# if __name__ == '__main__':
#     # compare_predictions()
#     model = example_k.model()
#     save_keras_model(model, freeze=True)
