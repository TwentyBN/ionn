from __future__ import (
    print_function,
    division,
    unicode_literals,
    absolute_import,
)

import os
import numpy as np
import tensorflow as tf

from ionn import tfpb


class TestLoading(tf.test.TestCase):

    def setUp(self):
        self.tmp = tf.test.get_temp_dir()
        self.fname = os.path.join(self.tmp, 'test.pb')
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def get_session(self):
        return self.test_session(graph=self.graph)

    def create_example(self):
        with self.graph.as_default():
            a = tf.constant([2.])
            x = tf.placeholder(tf.float32, name='input')
            y = tf.multiply(a, x, name='output')
        with self.session.as_default():
            tf.train.write_graph(
                self.session.graph_def, self.tmp, 'test.pb', as_text=False)
        return y

    def test_load_protobuf(self):
        y_created = self.create_example()
        y_loaded = list(tfpb.load_protobuf(
            self.fname,
            session=self.session).values())
        input_dict = {'input:0': [3.]}

        with self.session.as_default():
            direct_result = self.session.run(y_created, feed_dict=input_dict)
            loaded_result = self.session.run(y_loaded, feed_dict=input_dict)

        self.assertAlmostEqual(direct_result, loaded_result)


class TestFreezing(tf.test.TestCase):

    def setUp(self):
        self.tmp = tf.test.get_temp_dir()
        self.fname = os.path.join(self.tmp, 'test.pb')
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def get_session(self):
        return self.test_session(graph=self.graph)

    def create_graph_def(self, session, initialized=True):
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                a = tf.Variable(tf.zeros((1,)))
                x = tf.placeholder(tf.float32, name='input')
                tf.add(a, x, name='output')
                with self.session.as_default():
                    if initialized:
                        self.session.run(tfpb.initializer())
                    return self.session.graph.as_graph_def()

    def test_var2const_should_remove_variable_nodes(self):
        with self.session.as_default():
            graph_def = self.create_graph_def(self.session)

            constant_graph_def = tfpb.var2const(
                graph_def, ['output'],
                session=self.session)

        for node in constant_graph_def.node:
            self.assertNotEqual(node.op, 'Variable')

    def test_var2const_should_raise_if_no_weights(self):
        with self.session.as_default():
            graph_def = self.create_graph_def(self.session, initialized=False)

            with self.assertRaises(tfpb.NoWeightsError):
                tfpb.var2const(graph_def,
                               ['output'],
                               session=self.session)          

    def test_store_load_gives_identity(self):
        input_dict = {'input:0': [3.]}

        with self.session.as_default():
            graph_def = self.create_graph_def(self.session)
            tf.import_graph_def(graph_def, name='')
            result_before = self.session.run('output:0', feed_dict=input_dict)

            # Save freezed graph and load
            tfpb.save_protobuf(graph_def, self.fname, session=self.session)
            recovered = tfpb.load_protobuf(self.fname, session=self.session)

            result_after = self.session.run(recovered, feed_dict=input_dict)

        self.assertAlmostEqual(result_before, result_after['output:0'])

    def test_clear_devices_removes_all_device_specs(self):
        with self.session.as_default():
            graph_def = self.create_graph_def(self.session)

        # Make sure there is some device information
        self.assertTrue(any([len(node.device) for node in graph_def.node]))

        clear_graph_def = tfpb.clear_devices(graph_def)

        self.assertFalse(any([len(node.device)
                              for node in clear_graph_def.node]))


class TestVariablesAfterSaving(tf.test.TestCase):

    def setUp(self):
        self.tmp = tf.test.get_temp_dir()
        self.fname = os.path.join(self.tmp, 'test.protobuf')
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def build_constant_graph(self):
        with self.graph.as_default():
            x = tf.placeholder("float", name='Input')
            w = tf.constant(10., name='Factor')
            tf.multiply(w, x, name='Output')

    def build_variable_graph(self):
        with self.graph.as_default():
            x = tf.placeholder("float", name='Input')
            w = tf.Variable(np.array([1.]), name='Factor', dtype=tf.float32)
            tf.multiply(w, x, name='Output')
            self.session.run(tf.global_variables_initializer())

    def build_trained_graph(self):
        with self.graph.as_default():
                x = tf.placeholder("float", name='Input')
                z = tf.placeholder("float", name='Target')
                w = tf.Variable(np.array([1.]), name='Factor', dtype=tf.float32)
                y = tf.multiply(w, x, name='Output')
                cost = tf.pow(y - z, 2)
                optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
                self.session.run(tf.global_variables_initializer())
                for i in range(10):
                    self.session.run(optimizer, feed_dict={x: 1, z: 10})

    def save_graph_reload_and_check(self, graph_builder, expected_output):
        with self.session.as_default():
            graph_builder()
            tfpb.save_protobuf(self.session.graph.as_graph_def(),
                               self.fname,
                               output_nodes=('Output:0',),
                               freeze=True,
                               session=self.session)
            node = tfpb.load_protobuf(self.fname,
                                      output_nodes=('Output:0',),
                                      session=self.session)
            loaded_value = self.session.run(list(node.values())[0],
                                            feed_dict={'Input:0': 1.,
                                                       'Target:0': 10.})
        self.assertNotEqual(loaded_value, 10.)


if __name__ == '__main__':
    tf.test.main()
