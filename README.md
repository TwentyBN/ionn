# IONN io-operations for artificial neural networks

IONN provides functionality to store, load and freeze neural networks and
convert networks between different neural network frameworks. The current version provides

1. Storing, loading, freezing of tensorflow models in google protobuf files (submodule `tfpb`)
2. Dumping keras models as google protobuf files and loading them into a pure tensorflow environment (submodule `k2tf`)


## tfpb - freeze and store graphs

Tensorflow provides a graph freezing tool that works ok, but is hardly
documented and not particularly modular. The `tfpb` module provides a
simplified interface to storing frozen graphs. There are two main entrypoints,
`load_protobuf` and `save_protobuf`. Furthermore, you can directly call tfpb to
freeze stored graphs like this

    tf-freeze <input_graph_file_name> <output_file_name> <checkpoint_file_name>


## k2tf - From keras to tensorflow

Keras is nice if we want to quickly draft out a neural network architecture.
Unfortunately, it differs considerably in how it stores models and can
therefore not well co-exist with tensorflow infrastructure. k2tf supports
storing keras models in tensorflow protobuf files that can later be loaded
without keras. There are currently two drawbacks though:

1. Models have to be frozen, which isn't exactly desirable because most of
   tensorflow's strength is in tweaking the models during the learning phase.
2. Models have to be reloaded in a separate process to avoid confusion about
   the tensorflow graph.
