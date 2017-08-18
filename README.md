# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"
[image9]: ./examples/6.jpg "Traffic Sign 6"
[image10]: ./examples/7.jpg "Traffic Sign 7"
[image11]: ./examples/8.jpg "Traffic Sign 8"
[image12]: ./examples/predict_1.png "1-4 Predictions"
[image13]: ./examples/predict_2.png "5-6 Predictions"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup
Here is a link to my [project code](https://github.com/jfrattarola/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![histogram][image1]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data should to have mean zero and equal variance.

I also converted the images from RGB to Grayscale, reducing the dimensions from 32,32,3 to 32,32,1. This increased my training accuracy by 8%!


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture, slightly modified:
1. Input layer: (32,32,1)
2. Layer 1: convolution with weights normalized with a mean of 0 and standard deviation of 0.1. Biases set to 0. Stride of 1, no padding
 - output shape: 28,28,6
 - activation: ReLU
 - applied dropout on 25% of the activation output
3. Layer 2: Added pooling layer with stride of 2 and ksize of 2 (no padding)
 - output shape: 14,14,6
4. Layer 3: convolution with weights normalized with a mean of 0 and standard deviation of 0.1. Biases set to 0. Stride of 1, no padding
 - output shape: 10,10,16
 - activation: ReLU
 - applied dropout on 25% of the activation output
5. Layer 4: Added pooling layer with stride of 2 and ksize of 2 (no padding)
 - output shape: 5,5,16
6. Flatten output to vector of size 400
7. Layer 5: fully-connected layer with weights normalized with a mean of 0 and standard deviation of 0.1. Biases set to 0.
 - output vector length: 120
 - applied dropout of 25%
8. Layer 6: fully-connected layer with weights normalized with a mean of 0 and standard deviation of 0.1. Biases set to 0.
 - output vector length: 84
 - applied dropout of 25%
9. Layer 7: Output layer. Fully-connected layer with weights normalized with a mean of 0 and standard deviation of 0.1. Biases set to 0.
 - output vector length: 10 (logits)



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I am using Adam Optimizer, calculating the loss using the mean of cross entropy output. Batch size: 128 Epoch size: 30 Learning rate: 0.001 (I had intended for decaying learning rate, but that didn't seem to improve training accuracy) Model saved for further use without need for re-training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Originally, I ran Lenet without dropout and I achieved a 99% training accuracy, but a 91% test-validation accuracy. Dropout brought the accuracy up to 94.7%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose 8 German signs

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image12] 
![alt text][image13] 

The model was able to correctly guess 8 out of 8 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook. Here are the softmax figures...

```
TopKV2(values=array([[  1.00000000e+00,   1.52336016e-13,   1.94979374e-17,
          2.47758936e-18,   1.23624663e-19]], dtype=float32), indices=array([[13, 34, 12, 38, 40]], dtype=int32))
TopKV2(values=array([[  9.91013229e-01,   8.98637623e-03,   3.68736778e-07,
          7.43885531e-09,   3.86033532e-11]], dtype=float32), indices=array([[27, 18, 11, 24, 26]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   5.70349190e-09,   7.41681647e-11,
          3.38198913e-11,   2.78385117e-13]], dtype=float32), indices=array([[16, 40,  9,  7, 10]], dtype=int32))
TopKV2(values=array([[  9.99994397e-01,   5.59153750e-06,   5.63399460e-08,
          6.77337475e-09,   9.29481436e-10]], dtype=float32), indices=array([[29, 28, 22, 23, 20]], dtype=int32))
TopKV2(values=array([[  9.99849200e-01,   1.02684091e-04,   3.83870829e-05,
          5.32308013e-06,   1.26958503e-06]], dtype=float32), indices=array([[12,  9, 15, 13, 40]], dtype=int32))
TopKV2(values=array([[  9.99983430e-01,   1.65025358e-05,   3.17538920e-08,
          3.17031166e-08,   1.93922602e-08]], dtype=float32), indices=array([[33, 13, 10, 35,  1]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   4.18538071e-17,   6.84047451e-19,
          1.28179610e-19,   1.30514342e-20]], dtype=float32), indices=array([[25, 21, 31, 20, 23]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   4.80363020e-08,   1.79552906e-09,
          1.38324940e-09,   8.59255944e-10]], dtype=float32), indices=array([[14, 17,  3,  2, 33]], dtype=int32))
```
