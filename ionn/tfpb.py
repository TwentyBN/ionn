from __future__ import (
    print_function,
    division,
    unicode_literals,
    absolute_import,
)

import logging
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import graph_util

# This is a wrapper for compatibility with old style and new style tensorflow
initializer = getattr(
    tf,
    'global_variables_initializer',
    getattr(tf, 'initialize_all_variables')
)


class NoWeightsError(Exception):
    pass


def load_protobuf(fname, output_nodes=('output:0',), session=None):
    """Load binary protobuf and return graph

    Args:
        fname:
            name of the file from which to load the graph
        output_nodes:
            names of the node that should contain the output
            Note that tensorflow is a bit inconsistent about this. Sometimes,
            output nodes come as '<node_name>:<output_index>', sometimes only
            '<node_name>'. We try to convert between the two if possible.
        session:
            the session to use. Not that this function will only work in a
            context where you have set session.as_default()

    Returns:
        dictionary of nodes by name
    """
    graph_def = load_graph_def(fname, binary=True)

    tf.import_graph_def(graph_def, name='')
    return {node: session.graph.get_tensor_by_name(node)
            for node in output_nodes}


def save_protobuf(graph_def, fname, output_nodes=('output:0',),
                  freeze=True, session=None):
    """Save graph to protobuf file

    Args:
        graph_def:
            graph definition (e.g. from `session.graph.as_graph_def()`)
        fname:
            name of the output protobuf file
        output_nodes:
            sequence of output nodes to store.
    """
    if freeze:
        onodes = [nodename.split(':')[0] for nodename in output_nodes]
        graph_def = var2const(graph_def, onodes, session=session)
    with tf.gfile.GFile(fname, 'wb') as f:
        f.write(graph_def.SerializeToString())
    logging.info('%d ops in the final graph', len(graph_def.node))


def var2const(graph_def, output_nodes, session=None):
    """Convert all variable nodes to constant nodes

    Args:
        graph_def:
            graph definition with variable nodes
        output_nodes:
            list of nodes that will containt network output
        session:
            the session to use. Note that this function will only work in a
            context where you have set session.as_default()

    Returns:
        graph definition with no variable nodes
    """
    try:
        return graph_util.convert_variables_to_constants(
                            session, graph_def, output_nodes)
    except tf.errors.FailedPreconditionError as e:
        raise NoWeightsError(e.message + 'There is no weights!')


def load_graph_def(fname, binary=True):
    """Load graph definition from file

    Args:
        fname:
            name of the file from which to load the graph
        binary:
            is the graph binary?

    Returns:
        GraphDef object
    """
    mode = 'rb' if binary else 'r'
    with tf.gfile.FastGFile(fname, mode) as f:
        return parse_graph_def(f.read(), binary)


def parse_graph_def(buf, binary=True):
    """Handle binary and non-binary graph def parsing

    Args:
        buf:
            buffer containing the graph definition
        binary:
            is the buffer binary to text?
    """
    graph_def = tf.GraphDef()
    if binary:
        graph_def.ParseFromString(buf)
    else:
        text_format.Merge(buf.decode('utf-8'), graph_def)
    return graph_def


def clear_devices(graph_def):
    for node in graph_def.node:
        node.device = ""
    return graph_def


def main(
    output_file,
    checkpoint_file,
    output_node_names='output:0',
):

    session = tf.Session()
    '''
    Returns a context manager that makes this object the default session.
    '''
    with session.as_default():
        # Restore weights
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(session, checkpoint_file)

        graph_def = clear_devices(session.graph.as_graph_def())

        save_protobuf(graph_def,
                      output_file,
                      output_node_names,
                      session=session)
    session.close()

if __name__ == '__main__':
    main()
