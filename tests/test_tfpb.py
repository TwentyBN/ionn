from __future__ import (
    print_function,
    division,
    unicode_literals,
    absolute_import,
)

import os
import tensorflow as tf

from ionn import tfpb


class TestLoading(tf.test.TestCase):

    def setUp(self):
        self.tmp = tf.test.get_temp_dir()
        self.fname = os.path.join(self.tmp, 'test.pb')
        self.graph = tf.Graph()

    def get_session(self):
        return self.test_session(graph=self.graph)

    def create_example(self):
        with self.graph.as_default():
            a = tf.constant([2.])
            x = tf.placeholder(tf.float32, name='input')
            y = tf.multiply(a, x, name='output')
        with self.get_session() as session:
            session.run(tfpb.initializer())
            tf.train.write_graph(
                session.graph_def, self.tmp, 'test.pb', as_text=False)
        return y

    def test_load_protobuf(self):
        y_created = self.create_example()
        y_loaded = list(tfpb.load_protobuf(
            self.fname,
            session=self.get_session).values())
        input_dict = {'input:0': [3.]}

        with self.get_session() as session:
            direct_result = session.run(y_created, feed_dict=input_dict)
            loaded_result = session.run(y_loaded, feed_dict=input_dict)

        self.assertAlmostEqual(direct_result, loaded_result)


class TestFreezing(tf.test.TestCase):

    def setUp(self):
        self.tmp = tf.test.get_temp_dir()
        self.fname = os.path.join(self.tmp, 'test.pb')
        self.graph = tf.Graph()

    def get_session(self):
        return self.test_session(graph=self.graph)

    def create_graph_def(self):
        with self.graph.as_default():
            a = tf.Variable(tf.zeros((1,)))
            x = tf.placeholder(tf.float32, name='input')
            tf.add(a, x, name='output')
        with self.get_session() as session:
            session.run(tfpb.initializer())
            return session.graph.as_graph_def()

    def test_var2const_should_remove_variable_nodes(self):
        graph_def = self.create_graph_def()

        constant_graph_def = tfpb.var2const(
            graph_def, ['output'],
            session=self.get_session)

        for node in constant_graph_def.node:
            self.assertNotEqual(node.op, 'Variable')

    def test_store_load_gives_identity(self):
        input_dict = {'input:0': [3.]}
        graph_def = self.create_graph_def()
        with self.get_session() as session:
            tf.import_graph_def(graph_def, name='')
            session.run(tfpb.initializer())
            result_before = session.run('output:0', feed_dict=input_dict)

        # Save freezed graph and load
        tfpb.save_protobuf(graph_def, self.fname, session=self.get_session)
        recovered = tfpb.load_protobuf(self.fname, session=self.get_session)

        with self.get_session() as session:
            session.run(tfpb.initializer())
            result_after = session.run(recovered, feed_dict=input_dict)

        self.assertAlmostEqual(result_before, result_after['output:0'])

    def test_clear_devices_removes_all_device_specs(self):
        graph_def = self.create_graph_def()

        # Make sure there is some device information
        self.assertTrue(any([len(node.device) for node in graph_def.node]))

        clear_graph_def = tfpb.clear_devices(graph_def)

        self.assertFalse(any([len(node.device)
                              for node in clear_graph_def.node]))


if __name__ == '__main__':
    tf.test.main()
