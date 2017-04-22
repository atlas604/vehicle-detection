**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

**Histogram of Oriented Gradients (HOG)**

**1. Explain how (and identify where in your code) you extracted HOG features from the training images.**

I use a series of functions combined with their properties into a function called `extract_features()` (lesson_functions.py line 52) to extract HOG features from the training images.  The steps consists of computing the gradient image in x and y coordinates to a chosen color space, computing the gradient histograms, normalizing the process and flattening everything into a feature vector.  The main algorithm utilized is scikit-image's hog feature that does the processing.  

Examples of the hog feature images look like this:

<img src="./output_images/hog1.png">
<img src="./output_images/hog2.png">

**2. Explain how you settled on your final choice of HOG parameters.**

I increased the orientation value to have a greater distribution of gradient directions for slightly more classification.  It's difficult to visually identify the differences in the HOG feature image when you change the values, so my main decision criterion for the parameters was decided by the performance of the classifier I trained.  The hog images for cars clearly illustrates a shape resembling a car and for non-cars the directions of the gradients were more scattered and harder to identify. Having more hog channels should further define this concept.  After testing, it appears that the choice of color space has the biggest impact in the performance of the model and YCrCb seemed to perform the best. Tuning the other parameters had little to no impact on the performance and thus were left as the default values showcased in the lessons.  

These are the final parameters:

|    Feature     | Parameter   |
|----------------|-------------|
| color space    |  YCrCb      |
| orient         |  11         |
| pix per cell   |  8          |
| cell per block |  2          |  
| hog channel    |  ALL        |
| spatial size   |  16, 16     |
| hist bins      |  16         |
| spatial feat   |  True       |
| hist feat      |  True       |
| hog feat       |  True       |


**3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**

I used scikit-learn's Linear Support Vector Classification (LinearSVC) to train my classifier (svc.py line 65).

I did this by using the extract_features() function I mentioned earlier to extract and create arrays for feature vectors and label vectors of the dataset of the cars and non-cars.  I then split the arrays and randomized the data into training and test sets, and executed the LinearSVC function to train the model.

The end result indicated a 99% performance accuracy using the parameters above.  

**Sliding Window Search**

**1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?**


**2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?**

---

### Video Implementation

**1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)**

Here's a [link to my video result](./project_video_output.mp4)

**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

---

###Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**
