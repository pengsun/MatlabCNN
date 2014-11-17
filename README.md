MatlabCNN
=========

Matlab codes for 2D Convolutional Neural Network

Inspired by "https://github.com/rasmusbergpalm/DeepLearnToolbox" and "https://github.com/vlfeat/matconvnet", but aims at educational purpose. Provides carefully desined matlab class hierachy that helps one to understand the workflow of Convolutional Neural Network and Multi Layer Perceptron (MLP) by simply reading the code. 

## Summary:
* Basical layer (M-to-N transform): 
 * Full connection, Convolutional[1]
 * Average Pooling, Max Pooling
* Auxiliary layer: Local Response Normalization[2]
* Activation layer (pointwise transform): Sigmoid, Relu[2]
* Regularization: Dropout (implemented as pointwise transform), Max-norm constraint[3]
* Parameter updating: Stochastic Gradient Descent (mini-batch) with Momentum and Weight Decay[3]
* Loss: Least Square (for classification/regression), Softmax/Cross-entropy/Logistic-loss (for classification)
* Visualization: class-model and class-saliency map [4]

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

## References
 ```
 [1] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
 [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
 [3] Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012).
 [4] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." arXiv preprint arXiv:1312.6034 (2013).
 ```
