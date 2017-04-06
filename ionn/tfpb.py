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

    Returns:
        dictionary of nodes by name
    """
    graph_def = load_graph_def(fname, binary=True)

    with get_default_session(session) as sess:
        tf.import_graph_def(graph_def, name='')
        return {node: sess.graph.get_tensor_by_name(node)
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

    Returns:
        graph definition with no variable nodes
    """
    with get_default_session(session) as sess:
        return run_init_run(
            graph_util.convert_variables_to_constants,
            sess, graph_def, output_nodes)


def run_init_run(func, session, *args):
    """Run func(session, *args) with potentially uninitialized variables"""
    try:
        return func(session, *args)
    except tf.errors.FailedPreconditionError:
        # session.run(initializer())
        return func(session, *args)


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


def get_default_session(session):
    """A helper session that allows dependency injection for testing"""
    if session is None:
        return tf.Session()
    elif getattr(session, '__call__', False):
        return session()
    else:
        return session


def clear_devices(graph_def):
    for node in graph_def.node:
        node.device = ""
    return graph_def


def main(
    output_file,
    checkpoint_file,
    output_node_names='output:0',
):

    with tf.Session() as session:
        # Restore weights
        session.run(['save/restore_all'], {'save/Const:0': checkpoint_file})
        session.run(initializer())

        graph_def = clear_devices(session.graph.as_graph_def())

    save_protobuf(graph_def, output_file, output_node_names)


if __name__ == '__main__':
    main()
