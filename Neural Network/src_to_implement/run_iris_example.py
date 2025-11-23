from Optimization.Optimizers import Sgd
from Optimization.Loss import CrossEntropyLoss
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Layers.Helpers import IrisData
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    # data
    data = IrisData(batch_size=16)

    # optimizer + network
    optimizer = Sgd(learning_rate=0.01)
    net = NeuralNetwork(optimizer)
    net.data_layer = data
    net.loss_layer = CrossEntropyLoss()

    # architecture
    net.append_layer(FullyConnected(4, 16))
    net.append_layer(ReLU())
    net.append_layer(FullyConnected(16, 3))
    net.append_layer(SoftMax())

    # train
    net.train(iterations=500)
    print("Final loss:", net.loss[-1])

    # test
    x_test, y_test = data.get_test_set()
    preds = net.test(x_test)
    print("Predictions shape:", preds.shape)
