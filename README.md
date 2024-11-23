# NeuralNetwork-cpp

A from-scratch neural network library implemented in C++20. Certain core operations are accelerated using standard library multithreading to achieve CPU performance on a par with reference numpy implementations.

## Features

- Input, Dropout and Dense (Fully Connected) layers
- Relu, Softmax, Sigmoid and Linear activation layers
- Categorical Cross Entropy, Binary Cross Entropy, Mean Squared Error and Mean Absolute Error loss functions
- Categorical and Regression accuracy functions
- SGD, Adagrad, RMSProp and Adam optimizers
- Convenient `Model` class for constructing neural networks

## Usage

To compile the library:

```shell
mkdir build && cd build/
cmake -G Ninja -DBUILD_SAMPLE=ON -DBUILD_UNIT_TESTS=ON -DBUILD_INTEGRATION_TESTS=ON -DBUILD_SYSTEM_TESTS=OFF -DBUILD_BENCH=ON ..
ninja
```

The repo also contains a sample network for classifying images of clothing items. To run the sample OpenCV must be installed, you must download the [Fashion MNIST](https://www.kaggle.com/datasets/samihatna/fashion-mnist) dataset, and you need an image of a clothing item to test the network on. To build the sample, set the CMake option `-DBUILD_SAMPLE=ON`. Run as follows:

```shell
./sample/mnistsample ~/fashion_mnist_images ~/dress.png
```

Training for 15 epochs with a batch size of 128 (7020 iterations) yields an accuracy of 91.8%.

## Example Network

```cpp
Model<double, Accuracy::CategoricalAccuracy, Optimizers::Adam,
        Loss::CategoricalCrossEntropy>
m;

Accuracy::CategoricalAccuracy<double> acc;
m.setAccuracy(acc);

Optimizers::Adam<double> opt(0.001, 0.001);
m.setOptimizer(opt);

Loss::CategoricalCrossEntropy<double> loss;
m.setLoss(loss);

auto l1 = std::make_shared<Layers::DenseLayer<double>>(784, 128);
m.addLayer(l1);

auto a1 = std::make_shared<Activations::Relu<double>>();
m.addLayer(a1);

auto l2 = std::make_shared<Layers::DenseLayer<double>>(128, 128);
m.addLayer(l2);

auto a2 = std::make_shared<Activations::Relu<double>>();
m.addLayer(a2);

auto l3 = std::make_shared<Layers::DenseLayer<double>>(128, 10);
m.addLayer(l3);

auto a3 = std::make_shared<Activations::Softmax<double>>();
m.addLayer(a3);

m.prepare();

m.train(std::make_unique<Vec2d<double>>(xTrain),
        std::make_unique<Vec2d<double>>(yTrain),
        std::make_unique<Vec2d<double>>(xTest),
        std::make_unique<Vec2d<double>>(yTest), 10, 128);
```
