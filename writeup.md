#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[class_grid]: ./examples/classes.png "Classes"
[class_histogram]: ./examples/class_histogram.png "Class Histogram"
[normalized]: ./examples/normalized.png "Normalized"
[augmented]: ./examples/augmented.png "Augmented"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[sign1]: ./test_signs/01.jpg "Traffic Sign 1"
[sign2]: ./test_signs/02.jpg "Traffic Sign 2"
[sign3]: ./test_signs/03.jpg "Traffic Sign 3"
[sign4]: ./test_signs/04.jpg "Traffic Sign 4"
[sign5]: ./test_signs/05.jpg "Traffic Sign 5"
[sign6]: ./test_signs/06.jpg "Traffic Sign 6"
[sign7]: ./test_signs/07.jpg "Traffic Sign 7"
[sign8]: ./test_signs/08.jpg "Traffic Sign 8"
[softmax]: ./examples/softmax.png "Softmax"

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

####2. Include an exploratory visualization of the dataset.

I generated two plots that were helpful in visualizing the dataset.

First, a representative example of a member from each class in the dataset.

![alt text][class_grid]

Second, a histogram showing the number of members of each class.

![alt text][class_histogram]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Data Preprocessing

The first step of the data pipeline was setting up an image processing pipeline for the training, validation and test images. This consisted of a number of stages.

- Cropping to 24x24
- Conversion from unsigned 8-bit to float32 with range approximately -1.0 to 1.0
- Adjustment of the image values to account for the mean value of the entire set

Cropping to 24x24 seemed to improve training efficiency and speed. Most of the images in the dataset are not tightly cropped - there were usually several pixels of background that were not contributing to the signal. Additionally, these edges of the signs themselves did not seem to be particularly important - there were no signs that were similar except for the borders.

A number of other approaches were tested and found not to improve performance in the final design. Using HSV seemed not to help, and neither did using only the grayscale channel of the converted HSV image (see discussion of the zeroth convolution layer). Normalizing the individual channels to a single standard deviation also did not seem to help.

One possibility that was not tested was to use HSV but normalize only the V channel.

Here's a random sample of the normalized input:

![alt text][normalized]

#### Data Augmentation

Keras ImageDataGenerator was used to augment the training dataset. Augmentation took place using the cropped, preprocessed images because Keras doesn't appear to work properly with the original 8-bit images.

The transformations applied were:

- Random rotation by up to 10 degrees
- Random shift by up to 10%
- Random zoom by up to 10%

There are a number of other transformations that were not tested but that could be easily applied. In
particular, varying brightness and adding noise could potentially be effective.

Initially data augmentation was applied across the entire training set: typically 32k or 64k additional
images were added, based on a random selection of the entire dataset.

At a later stage, class-based augmentation was implemented: for each class, a certain number of augmented samples were generated in order to bring the number of members of that set to a specific number. With the current model, a baseline of 800 members per class was set. A larger baseline was tested but did not
seem to improve performance.

Here's a random sample of the augmented input:

![alt text][augmented]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model architecture was similar to the LeNet architecture with a few additions:

  - An initial 1x1x3 convolution layer was added in order to perform dynamic mapping from RGB into a single-channel image.
  - Dropout was added to the fully connected layers
  - An additional branch was added, bringing the original first conv layer to the first fully
  connected layer after a max pooling step.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 24x24x3 RGB image   							| 
| Convolution 1x1x3     | Outputs a 24x24x1 image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 20x20x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64    				|
| Fully connected		| input 7296, output 64     					|
| RELU					|												|
| Dropout				| 0.85											|
| Fully connected		| input 64, output 200    						|
| RELU					|												|
| Dropout				| 0.85											|
| Fully connected		| input 200, output 43    						|
| RELU					|												|
| Softmax				|         								    	|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training was performed using the Adam optimizer, a learning rate of 0.001, and epochs ranging
from 10 to 20. The dropout and sizes of all layers were systematically varied, as well as the
input pipeline and data augmentation parameters.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In all, over 90 variations were tested. For the purposes of this project, emphasis was placed on
exploring the parameter space over exhaustive training; a high priority was placed on the running
time of each training cycle, with most tests taking 5-10 minutes. 

A brief description of some of the variations:

**RGB vs. HSV vs. Grayscale**

In general, without the 1x1 convolution layer, RGB and Grayscale seemed to perform approximately as well if controlling for training time and total network size - Grayscale performed worse but much faster at first, and increasing the size of some layers and training epochs seemed to narrow the gap. HSV usually performed worse.

After adding the 1x1 conv layer, the RGB performance seemed to improve past Grayscale, with relatively small differences in training time.

**Image Normalization**

A number of types of image normalization were tested, and the one that seemed to help was to zero according to the global mean of the training set. Normalizing to a standard deviation did not seem to help.

**Image Augmentation**

After a large number of attempts where image augmentation seemed to lowering performance significantly (probably because the Keras implementation doesn't seem to work properly with 8-bit unsigned RGB values), the augmentation pipeline was placed after normalization and seemed to help. Initially, global augmentation by 32k or 64k images was tested, and seemed to help prevent overfitting. Later, class-based augmentation was applied and a modest amount (augmentation up to a minimum of 800 samples per class) seemed to perform better than larger amounts.

**Branch from Conv1 to FC1**

The branch from the first main convolution layer to the first fully connected layer (suggested by the LeCun paper) seemed to help when using RGB, but didn't seem to help when using Grayscale. 

**Dropout**

Dropout was periodically varied between 0.5 and 0.85. Higher levels of dropout seemed to prevent overfitting when very large fully connected layers were used, but seemed to hurt performance with smaller layers.

**Layer Sizes**

A variety of layer sizes were explored. In general, very small sizes (less than 32 neurons per layer) provided poor performance. For convolutional layers, 32 and 64 seemed to provide a good compromise between training time and performance; for fully connected layers 64 and 200 seemed to work well. Much larger sizes (up to 1024) were tested but did not appear to perform much better, especially given the required training time.

**Additional Layers**

Additional convolutional and fully connected layers were tested but did not seem to immediately improve performance. It's likely that these could improve performance with a larger simultaneous variation in layer sizes.

**Final Results

* Validation set accuracy of ~0.975
* Test set accuracy of ~0.960

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] ![alt text][sign4]
![alt text][sign5] ![alt text][sign6] ![alt text][sign7] ![alt text][sign8]

These are fairly high quality, well lit images that have been cropped similar to the training
set, so it is expected that the model should perform reasonably well.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| 30 km/h     			| 30 km/h 										|
| Turn left ahead	    | Turn left ahead								|
| 50 km/h	      		| 50 km/h					 			    	|
| Roundabout			| Roundabout          							|
| Road Work			    | Road Work          							|
| Priority Road			| Priority Road        							|
| Ahead Only			| Ahead Only          							|

The model was able to predict 8 out of 8, for an accuracy of 100%. This compares to the 96% accuracy
on the test set.



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is very good at predicting the more distinct signs such as No Entry or the large arrows. There was some ambiguity on the speed number signs - the "30 km/h" sign also has a reasonably strong secondary signal as a "80 km/h" sign as well as a number of others, the "50 km/h" sign also to a lesser extent. The "Road Work" sign shows some similarity with "Road narrows on the right".

It's very possible that the model's performance on some of these more ambigious signs could improve after longer training - 10 epochs is probably fine for the simpler signs but is barely adequate for the more challenging ones.

![alt text][softmax] 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In general, the network used the large primary central feature, ignoring details such as the shape of the sign and the borders.

**See the notebook for additional visuals and comments**


