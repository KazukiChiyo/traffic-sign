# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./outputs/by_class.png "Plots by class before augmentation"
[image2]: ./outputs/statistics.png "Histogram by class"
[image3]: ./outputs/transformation.png "Transformations"
[image4]: ./outputs/by_class_augment.png "Plots by class after augmentation"
[image5]: ./outputs/enhance.png "After enhancing"
[image6]: cnn.png "CNN architecture"
[image7]: incptv3.png "Inception layer architecture"
[image8]: ./outputs/test_imgs.png "Test images"
[image9]: ./outputs/test_imgs_enhanced.png "Test images enhanced"
[image10]: ./outputs/test_imgs_pred.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KazukiChiyo/traffic-sign/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set before data augmentation is 34799.
* ... after data augmentation is 129000.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I plotted 10 images from each class:

![alt text][image1]

This is a histogram showing the distribution of dataset by class:

![alt text][image2]

Since the number of training examples by each class is unbalanced, I used data augmentation to augment the training set in the following way: Each class is augmented to 3000 images and each generated image transforms an existing image of the same class by one of the three metrics:

* Random rotation with degree of rotation randomly drawn from range (-25.0, 25.0);
* Random noise with variance = 0.0005;
* Random shear with affine transformation from range (0.0, 0.1).

![alt text][image3]

This way I am able to get a more balanced dataset. Below is a visualization of the transformed images by class:

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I transformed the images to grayscale, then normalize the images and multiply the resulting array with a number slightly greater than 1; this way I am able to "brighten" the images. Below is a comparison between the images before enhancing and after enhancing:

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

*All layers uses strides 1x1 and valid padding.*

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   					|
| Convolution 1x1     	| output: 32x32x1, activation: tanh             |
| Convolution 5x5       | output: 28x28x32, activation: tanh            |
| Convolution 5x5       | output: 24x24x64, activation: tanh            |
| Max pooling	      	| 2x2 stride,  outputs 12x12x64 				|
| Convolution 3x3	    | output: 10x210x64, activation: tanh           |
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Inception V3@32	    | activation: tanh 			                    |
| Flat	                | output: 800 			                        |
| Dense	                | output: 1024 			                        |
| Dense	                | output: 256 			                        |
| Dropout	            | keep_prob: 0.5     	                        |
| Dense	                | output: 43     	                            |


Below is a Alex-Net style diagram of my architecture:

![alt text][image6]

Architecture of the inception layer (reference: https://datascience.stackexchange.com/questions/15328/what-is-the-difference-between-inception-v2-and-inception-v3):

![alt text][image7]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used Adam optimizer to train the model. Learning rate is 0.0001, batch size is 128, number of epochs is 10.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.986.
* test set accuracy of 0.970.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on Wikipedia:
![alt text][image8]

The quality of the testing images is significantly better then those the model had seen in the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The testing images are scaled down to 32x32x3; also, they have to be enhanced the same way as we did before:

![alt text][image9]

Here are the results of the prediction:

![alt text][image10]


The model was able to correctly classify 11 of the 12 traffic signs, which gives an accuracy of 91.667%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The result for each images can be found below:

```
Sign 1
No entry 1.0
Speed limit (20km/h) 0.0
Speed limit (30km/h) 0.0

Sign 2
Stop 0.99987936
Yield 7.8411016e-05
Speed limit (60km/h) 3.4378656e-05

Sign 3
Keep left 1.0
Speed limit (50km/h) 2.2929331e-08
Speed limit (100km/h) 8.116507e-09

Sign 4
Right-of-way at the next intersection 0.68062425
Roundabout mandatory 0.23523653
Traffic signals 0.06066228

Sign 5
Children crossing 1.0
Bicycles crossing 4.4072153e-28
Beware of ice/snow 7.899158e-32

Sign 6
General caution 1.0
Traffic signals 1.456567e-17
Pedestrians 1.1574889e-20

Sign 7
Double curve 0.9999976
Beware of ice/snow 2.1213336e-06
Right-of-way at the next intersection 2.5184386e-07

Sign 8
Right-of-way at the next intersection 1.0
Beware of ice/snow 6.7663626e-09
Children crossing 1.1142001e-12

Sign 9
Road narrows on the right 0.9995833
General caution 0.0003745716
Traffic signals 4.2114032e-05

Sign 10
End of all speed and passing limits 0.999376
Keep right 0.00062400085
Priority road 1.0969156e-11

Sign 11
Priority road 1.0
Yield 6.476929e-18
No vehicles 1.4422723e-20

Sign 12
Speed limit (60km/h) 0.9997712
Speed limit (50km/h) 0.0001196688
Wild animals crossing 5.616921e-05

```

`Sign 4` is the image the model fails to correctly classify, and from the probabilities statistics, we have a relatively low confidence of classifying the image as a "Right-of-way at the next intersection" (68.062425%).
