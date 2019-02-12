import os
import struct

import numpy as np


def load_train_data(num_samples=0, mode="numpy"):
    return load_mnist(num_samples, "train", mode=mode)


def load_test_data(num_samples=0, mode="numpy"):
    return load_mnist(num_samples, "test", mode=mode)


def load_mnist(num_samples, target, mode="numpy"):
    if mode != "numpy" and mode != "list":
        raise ValueError("Invalid return mode")

    os.chdir("Resources")
    print("Loading MNIST " + target + ": {} samples".format("all" if num_samples == 0 else num_samples))

    print("Loading labels")
    with open(os.path.join(os.getcwd(), target + "-labels.ubyte"), "rb") as f:
        byte = f.read(4)
        magic = struct.unpack(">I", bytearray(byte))[0]
        if magic != 2049:
            raise IOError("Wrong magic number {}. It should be 2049. Check your files!".format(magic))

        byte = f.read(4)
        size = struct.unpack(">I", bytearray(byte))[0]
        num_samples = size if num_samples == 0 else np.min([num_samples, size])
        byte = f.read(num_samples)
        _labels = []
        for b in bytearray(byte):
            aux = np.zeros(10)
            aux[b] = 1
            _labels.append(aux)

    print("{} labels loaded".format(num_samples))

    print("Loading images")
    with open(os.path.join(os.getcwd(), target + "-images.ubyte"), "rb") as f:
        byte = f.read(4)
        magic = struct.unpack(">I", bytearray(byte))[0]
        if magic != 2051:
            raise IOError("Wrong magic number {}. It should be 2051. Check your files!".format(magic))

        byte = f.read(4)
        size = struct.unpack(">I", bytearray(byte))[0]
        num_samples = size if num_samples == 0 else np.min([num_samples, size])

        byte = f.read(4)
        num_rows = struct.unpack(">I", bytearray(byte))[0]
        byte = f.read(4)
        num_cols = struct.unpack(">I", bytearray(byte))[0]

        _images = [np.array(bytearray(f.read(num_rows * num_cols))).reshape(784) / 255 for _ in range(num_samples)]

    print("{} images loaded".format(num_samples))

    os.chdir("..")

    if mode == "numpy":
        return np.array(_images).T, np.array(_labels).T
    elif mode == "list":
        return _images, _labels
