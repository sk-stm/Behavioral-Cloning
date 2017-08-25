#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./3_laps_data/center_2017_08_25_14_22_57_185.jpg "Center IMage example"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 6 and 16. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and mirrored data to reduce the bias of the track having many left turns

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple one and add more data to it until it unterfits.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it was appropriate for the last project. Even though I had not much hope beacause this task seemed more diverse to me.

TODO

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a ... mean squared error on the training set but a ... mean squared error on the validation set. This implied that the model was ... 

To combat the ..., I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Especially on right turns. To improve the driving behavior in these cases, I added the mirrored data set to the training data to equally let the car learn left and right turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

TODO
The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 
(note: visualizing the architecture is optional according to the project rubric)
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on the track if it once got off. These images show what a recovery looks like starting from the right to the left:

TODO
![alt text][image3]
![alt text][image4]
![alt text][image5]


TODO
Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would generalize even more.

After the collection process, I had 16280 number of data points. I then preprocessed this data by normalizing each pixel.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss of training and validation set that did not decreasy any further. I used an adam optimizer. Manually training the learning rate wasn't necessary.
