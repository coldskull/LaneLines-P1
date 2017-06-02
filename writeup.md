# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image2]: ./test_images_output/solidWhiteRight.png "Original"

[image3]: ./test_images_output/solidWhiteRight_canny_plus_hough.png "Canny + Region of Interest + Hough"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline has two main parts: 
1. Image processing (orig->grayscale->canny->region_masking->houghlines)
2. Calculate and draw a fitted line on current image using:
    * hough lines for current image
    * Fitted lines from previous images
    * Applying region of interest mask to trim the lines drawn.

![alt text][image2]
![alt text][image3]


### 2. Identify potential shortcomings with your current pipeline


A few obvious shortcomings of this pipeline are:
* Sensitive to hand-tuned 'region of interest' mask
* not able to deal with Curves because of line fitting and naieve historic averaging


### 3. Suggest possible improvements to your pipeline

A few possible improvements that i can think off are:
* fit a polynomial instead of a line so that curves can be better handled
* apply specialized canny which would only use derivatives on horizontal axis
* Make all functions generic and not assume image shape in any part of the code.
