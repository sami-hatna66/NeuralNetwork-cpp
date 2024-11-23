#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Model.hpp"
#include "test_helpers.hpp"

std::pair<Vec2d<double>, Vec2d<double>>
loadMnist(const std::string &datasetPath, const std::string &&subdataset) {
    Vec2d<double> X;
    Vec2d<double> y;
    y.push_back({});
    for (const auto &clothingFolder :
         std::filesystem::directory_iterator(datasetPath + "/" + subdataset)) {
        if (std::filesystem::is_directory(clothingFolder.status())) {
            for (const auto &clothingImg :
                 std::filesystem::directory_iterator(clothingFolder.path())) {
                cv::Mat readImg =
                    cv::imread(clothingImg.path(), cv::IMREAD_GRAYSCALE);
                readImg =
                    readImg.reshape(1, readImg.total() * readImg.channels());
                std::vector<double> imgVec =
                    readImg.isContinuous() ? readImg : readImg.clone();
                X.push_back(imgVec);
                y[0].push_back(
                    std::stod(clothingFolder.path().filename().string()));
            }
        }
    }
    return {X, y};
}

TEST(SystemTests, FashionMnist) {
    std::cout << "Loading train data ..." << std::endl;
    auto [xTrain, yTrain] = loadMnist("./test/fashion-mnist/fashion_mnist_images", "train");
    std::cout << "Loading test data ..." << std::endl;
    auto [xTest, yTest] = loadMnist("./test/fashion-mnist/fashion_mnist_images", "test");

    std::cout << "Pre-processing data ..." << std::endl;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 xEng(seed);
    auto yEng = xEng;
    std::shuffle(xTrain.begin(), xTrain.end(), xEng);
    std::shuffle(yTrain[0].begin(), yTrain[0].end(), yEng);

    xTrain = (xTrain - 127.5) / 127.5;
    xTest = (xTest - 127.5) / 127.5;

    std::cout << "Constructing model ..." << std::endl;

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
            std::make_unique<Vec2d<double>>(yTest), 15, 128);

    std::map<int, std::string> labels{
        {0, "T-shirt/Top"}, {1, "Trouser"},    {2, "Pullover"}, {3, "Dress"},
        {4, "Coat"},        {5, "Sandal"},     {6, "Shirt"},    {7, "Sneaker"},
        {8, "Bag"},         {9, "Ankle Boot"},
    };

    EXPECT_TRUE(m.getAccuracy() >= 0.8);
}