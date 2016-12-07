# IONN io-operations for artificial neural networks

IONN provides functionality to store, load and freeze neural networks and
convert networks between different neural network frameworks. The current version provides

1. Storing, loading, freezing of tensorflow models in google protobuf files (submodule `tfpb`)
2. Dumping keras models as google protobuf files and loading them into a pure tensorflow environment
