## Exercise 1 – Neural Network from Scratch (NumPy)

This project implements a small, layer-oriented neural network framework in NumPy as part of the FAU *Deep Learning* course.

### Implemented components

- **Sgd optimizer**  
  - Basic stochastic gradient descent: `w_new = w_old - lr * grad`.

- **Layers**
  - `BaseLayer` – base class, manages `trainable` flag.
  - `FullyConnected` – linear layer with bias, weight updates via optimizer.
  - `ReLU` – non-linear activation, element-wise `max(0, x)`.
  - `SoftMax` – converts logits to class probabilities.

- **Loss**
  - `CrossEntropyLoss` – forward + backward, accumulated over batch.

- **NeuralNetwork**
  - Holds `layers`, `data_layer`, `loss_layer`, and `optimizer`.
  - Implements `forward()`, `backward()`, `train(iterations)`, and `test(input_tensor)`.

### Testing

All components are validated with the provided unit test suite:

```bash
python NeuralNetworkTests.py
python NeuralNetworkTests.py Bonus
