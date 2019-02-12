import numpy as np
import MNIST.mnist_loader as loader
import Net.network as nn

def main():
    net = nn.Network([28 * 28, 30, 10])
    x, y = loader.load_train_data()
    print(x.shape)
    print(net.feed_forward(x))

if __name__ == '__main__':
    main()
