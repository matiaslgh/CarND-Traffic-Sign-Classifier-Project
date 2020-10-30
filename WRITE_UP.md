# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

[//]: # "Image References"
[data-distribution]: ./examples/data-distribution.png "Data Distribution"
[some-images]: ./examples/some-images.png "Some Images"
[alpha-beta]: ./examples/alpha-beta.png "Function for alpha/beta parameters"
[contrast-brightness]: ./examples/contrast-brightness.png "Contrast/Brightness improvements"
[grayscale]: ./examples/grayscale.png "Grayscale Image"
[rotated]: ./examples/rotated.png "Rotated Image"
[data-augmentation]: ./examples/data-augmentation.png "Data augmentation"
[softmax-results]: ./examples/softmax-results.png "Softmax results"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

### Data Set Summary & Exploration

#### 1. Basic summary

- The size of training set is 34799
- The size of the validation set is 4410
- The size of test set is 12630
- The shape of a traffic sign image is (32, 32, 3)
- The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

#### **Data Distribution**

We have unbalanced data and that will affect the predictions negatively
![alt text][data-distribution]

#### **Some images in the training dataset**

![alt text][some-images]

### Design and Test a Model Architecture

#### 1. Preprocessing + Data augmentation

**Improve brightness and contrast**

We can use this opencv's function cv2.convertScaleAbs(image, alpha, beta) to improve brightness and contrast. It basically modifies every pixel by doing pixel \* alpha + beta. More info here The main challange here is that some images have a great contrast/brightness and modifying them makes results worse, so I procedeed to make a function that returns the value of alpha/beta based on an input X. What's that X? well, I noticed that the average between every pixel's value of a dark image is lower than the one for a brighter image, so that's the value I'll use as input for my get_alpha and get_beta functions. Then, I just had to set the limits and the restrictions to make the results lower when the input was higher.

![alt text][alpha-beta]

Here we can see the comparison between the original image and the one with contrast/brightness improvements. Some of them remain without any change because they look well as they are, but others have these improvements. The comparison is between rows (1-2, 3-4, ,5-6)

![alt text][contrast-brightness]

**Convert to grayscale**

I noticed that we don't really need to see colors to identify a traffic sign, so I decided to convert the images to grayscale in order to simplify the input to reduce complexity, improve performance and likely increase accuracy

![alt text][grayscale]

**Data augmentation**

First I rotated every image of the dataset of training 10 degrees in both directions. This tripled the amount of images. These small changes create "new images" for training and help the model to learn better the content by not depending only on a fixed position/orientation.

![alt text][rotated]

Then, the idea was to apply every of the transformations below with the goal of forcing the model to learn to classify an image with a partial view (just like humans could do it) but I couldn't apply all of them because I ran out of memory. As a future improvement I should improve the way I do this, since data augmentation is one of the things that improved accuracy the most

![alt text][data-augmentation]

**Normalization**

I normalized the images by doing `(pixel - 128)/ 128` to convert the int values of each pixel [0,255] to float values with range [-1,1]. This reduces the impact of each pixel in a multiplication and also helps by setting the mean close to 0.

#### 2. Model Architecture

Model architecture based on LeNet

| Layer             | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| Input             | 32x32x1 grayscale image                                             |
| Convolution       | ksize=5x5 / stride=1x1 / valid padding / depth=32 / output=28x28x32 |
| RELU              |                                                                     |
| Max pooling       | ksize=2x2 / stride=2x2 / valid padding / output=14x14x32            |
| Convolution       | ksize=5x5 / stride=1x1 / valid padding / depth=16 / output=10x10x16 |
| RELU              |                                                                     |
| Max pooling       | ksize=2x2 / stride=2x2 / valid padding / output=5x5x16              |
| Flatten           | outputs 400                                                         |
| Dropout           | keep_prob=0.75                                                      |
| Fully connected   | outputs 120                                                         |
| RELU              |                                                                     |
| Dropout           | keep_prob=0.75                                                      |
| Fully connected   | outputs 84                                                          |
| RELU              |                                                                     |
| Dropout           | keep_prob=0.75                                                      |
| Fully connected   | outputs 43                                                          |
| Softmax           |                                                                     |

#### 3. Training

I used Adam Optimizer to minimize the mean of the cross entropy between the predictions of my model and the actual labels.

These are the hyperparameters I used:

* learning rate = 0.001
* dropout (prob to keep) = 0.75
* epochs = 10
* batch size = 128
* weights initialization: truncated_normal with mu = 0 and sigma = 0.1

My results:

* validation set accuracy of 97.7%
* test set accuracy of 96.4%

#### 4. Solution approach

This was my iterative process:

1. I used orginal LeNet modifying the input to 32x32x**3** and I normalized the data by applying `(pixel - 128)/ 128` to every image. `Validation accuracy: 0.684`
2. I improved the brightness and contrast because I noticed that some images were almost impossible to classify even by a human. `Validation accuracy: 0.717`
3. I realized that humans could classify perfectly the traffic signs even if they were in grayscale. So I converted the images to grayscale to reduce the size and complexity of the trained model. `Validation accuracy: 0.755`
4. I applied data augmentation by rotating the images or hiding some sections. Since the way I wanted to do this wasn't super smart, I ran out of memory (yeah, I tried to load everything in RAM). So I only applied these transformations: rotations of +/- 10 degrees. This increased the training dataset size from X to **X + 2X**. Then, to all of them I removed corners: top right, top left, bottom right and bottom left. This increased the entire size 4 times --> **(X + 2X) \* 4**. `Validation accuracy: 0.951`
5. I added dropout before every fully connected layer with a keep_prob of 0.75. `Validation accuracy: 0.968`
6. I increased the first filter depth from 6 to 32. `Validation accuracy: 0.977`

### Test a Model on New Images

#### 1. Acquiring New Images

Well I know I had to get 5 images, but I wanted to know how my model behaved with a larger dataset of random unknown traffic signs. That's why I got some from google images, some mockups from [wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany) and finally I spent a while copying/cropping traffic signs from [this youtube video](https://www.youtube.com/watch?v=TGzL1Z3INnw)

As a result, I built a dataset of 104 images :D . You will see them in the final analysis.

I wanted to know how well the model would behave with mockups, since I didn't use that kind of images to train it. That was the first challenge I had in mind. Then I detected 3 more things that could be hard to classify by the model: light reflection, angle of the picture and tricky contrast

#### 2. Performance on New Images

82 out of 104 images were well classified. That gives us: **`Accuracy = 0.788`**

I'm really happy with this result as a first try and after the analysis shown below I have some ideas to increase this accuracy :)

#### 3. Model Certainty - Softmax Probabilities

I'm sorry for the length of this image, but I think it's full of useful information.

![alt text][softmax-results]

**Some conclusions**

Well to provide better information I should get some extra metrics like level of certainty when it classifies correctly a class, level of certainty when it misclassifies a class, which classes the model mixes up and so on, but roughly I can detect that it tends to classify better the classes that had more data in the training set (unbalanced data problem) and usually when it misclassifies a class that had few samples in the training data it is with a class that had more samples. This should be fixed by dealing with the problem of unbalanced data.

In some cases, even when the image was well classified, we can see the level of confidence in the prediction says that the model was just lucky. This is something to improve as well.

Some interesting points:

- It mixes `children crossing` with `bikes crossing` (this is conceptually funny to me :P)
- It struggles to differentiate the "max speed" signs between different limits.
- I put a speed limit sign that was made with LEDs, that's why it has different colors. Obviously, it fails to classify it correctly. I could use color transformations to make more data augmentation when training
- Some normal success cases: `yield`, `priority road`, `no entry`, `keep right`
- Some normal failure cases: `pedestrians`, `road narrows on the right`, `children crossing`

#### 4. Future improvements

1. Deal with unbalanced data
2. Think a smarter way to augmentate data by avoiding having everything in RAM
3. Add more ways to augmentate the data (like color transformation)
4. Get metrics of confidence on misclassification, confidence on correct classifications, how it misclassifies, and so on to be able to detect the model's flaws
5. Visualize the Neural Network's State with Test Images
