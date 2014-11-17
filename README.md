MatlabCNN
=========

Matlab codes for 2D Convolutional Neural Network

Inspired by "https://github.com/rasmusbergpalm/DeepLearnToolbox" and "https://github.com/vlfeat/matconvnet", but aims at educational purpose. Provides carefully desined matlab class hierachy that helps one to understand the workflow of Convolutional Neural Network and Multi Layer Perceptron (MLP) by simply reading the code. 

## Summary:
* Basical layer (M-to-N transform): 
 * Full connection, Convolutional
 * Average Pooling, Max Pooling
* Auxiliary layer: Local Response Normalization
* Activation layer (pointwise transform): Sigmoid, Relu
* Regularization: Dropout (implemented as pointwise transform), Max-norm constraint
* Parameter updating: Stochastic Gradient Descent (mini-batch) with Momentum and Weight Decay
* Loss: Least Square (for classification/regression), Softmax/Cross-entropy/Logistic-loss (for classification)
* Visualization: class-model and class-saliency map 

**Caution: Feel free to use the code, but it is primarily for my personal playing around and the developement is ongoing, so no guarantee for bug-free:)**

## TODO
### code
* myCNN
 * Display loss 
 * Continue training from loaded model
* trans 
 * Convolutional layer picking random subset of input feature maps
* Maxout?

### doc
* A note giving more mathematical details than those in "Notes on Convolutional Neural Networks" by Jake Bouvrie
  * Multi-dimensional array calculus
  * Derivative for Convolution; Convolution as sum of convoluation of pulses
* A design manual explaining why the "atomic layer"
  * "Atomic layer" as transformation: M-to-N transform and M-to-M (point-wise) transform
  * Chain rule as Back Propagation
  * Dropout as a point-wise transform
