
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

# ## Import Packages

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[2]:


    
#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

#(540, 960, 3)
# This defines a trapezoidal area of interest for 540X960 image
trapezoid = np.array([ [100,540], [450,325], [550,325], [900,540]], np.int32)


#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[6]:

import math

def show_image(img, text):
    """Displays an image
    This will display and image with
    a caption and its type and shape
    
    'img' is the image
    'text' is the caption """
    print(text , 'Type is:', type(img), 'with dimensions:', img.shape)
    plt.imshow(img)

def apply_canny_and_hough(image, trapezoid):
    """Finds raw line segment
    This is the core helper function
    of the video pipeline. It applies canny/hough
    as well as filters out the area outside of region of interest
    
    'image' input image to be processed
    'trapezoid' a numpy array with list of x,y coordinates of masking region
    
    Returns hough lines as well as orig image with hough lines drawn on it"""
    
    #convert to grayscale before applying canny
    gray_image = grayscale(image)
    #set pixels lower than threshold to black
    gray_image[gray_image < 115] = 0
    
    #now mask off any pixel not inside area of interest
    gray_image = region_of_interest(gray_image, [trapezoid])
    
    #gaussian blur plus canny
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_image,(kernel_size, kernel_size),0)
    canny_image = canny(blur_gray, 20, 50)
    
    #hough lines
    rho = 1
    theta = np.pi/180
    threshold = 3
    min_line_len = 15
    max_line_gap = 10
    [lines,hough_image] = hough_lines(canny_image, rho, theta, threshold, min_line_len, max_line_gap)
    return [lines,hough_image] 
    
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print(type(lines), type(lines[0]))
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return [lines,line_img]

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.6, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[7]:

show_image(image, 'Original Image')


# In[8]:

[lines,hough_image] = apply_canny_and_hough(image,trapezoid)

show_image(hough_image,"hough")


# In[9]:

import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[10]:

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[11]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#glabal list to save coordinates of drawn lines for last N frames
lines_left = []
lines_right = []


# In[12]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    """
    `image` is the input image
    
    The result image is returned with lines annotated 

    """
    #glabal list to save coordinates of drawn lines for last N frames
    global lines_left, lines_right
    
    
    #get hough lines
    [lines,hough_image] = apply_canny_and_hough(image,trapezoid)
    
  
    
    #blank image (all black) on which we will draw lane lines
    final_image = np.zeros((540,960,3), np.uint8)

    
    #number of lines from previous frames that are already in the list
    history_len = len(lines_left)
    #print("history length at entry ", len(lines_left))
    
    for line in lines:
        
        for x1,y1,x2,y2 in line:
            if(x2 == x1): #skip lines that are vertical
                continue
            
            slope = (y2-y1)/(x2-x1)
            c = y1 - (slope*x1)
            xcept = (540-c)/slope
            
            if(xcept > 900 or xcept < 100): #skip lines that have x-intercept beyond region of interest
                continue
            
            if(x1 > 450 and x2 > 450): #lines with x coordinate greater than 450 are considered part of right lane
                lines_right += [x1,y1]
                lines_right += [x2,y2]
            elif(x1 < 450 and x2 < 450): #lines with x coordinate less than 450 are considered part of left lane
                lines_left += [x1,y1]
                lines_left += [x2,y2]
            else:
                #print("!!!!!! ", x1, x2, y1, y2) #skip ambiguous line (neither left nor right)
                continue

    #print("history length at mid ", len(lines_left))
    
    #convert list to np array
    lines_left_shaped = np.array(lines_left).reshape(len(lines_left)//2, 2)
    lines_right_shaped = np.array(lines_right).reshape(len(lines_right)//2, 2)
    
    #trim list of lines(coordinates) stored to max of 20 (equivalent to 5 frames)
    if(history_len < 20):
        lines_left = lines_left[:history_len]
        lines_right = lines_right[:history_len]
    else:
        lines_left = lines_left[4:20]
        lines_right = lines_right[4:20]
    
    #fit the left lane line and draw it
    vx,vy,cx,cy = cv2.fitLine(lines_left_shaped, cv2.DIST_L2,0, 0.01, 0.01)
    cv2.line(final_image, (int(cx-vx*350), int(cy-vy*350)), (int(cx+vx*350), int(cy+vy*350)), [255,0,0], 7)
    
    #save the coordinates so the next frame can use this history
    lines_left += [int(cx-vx*350),int(cy-vy*350)]
    lines_left += [int(cx+vx*350),int(cy+vy*350)]
    
    #fit the right lane line and draw it
    vx,vy,cx,cy = cv2.fitLine(lines_right_shaped, cv2.DIST_L2,0, 0.01, 0.01)
    cv2.line(final_image, (int(cx+vx*350), int(cy+vy*350)), (int(cx-vx*350), int(cy-vy*350)), [255,0,0], 7)
    
    #save the coordinates so the next frame can use this history
    lines_right += [int(cx+vx*350),int(cy+vy*350)]
    lines_right += [int(cx-vx*350),int(cy-vy*350)]
    
    #apply region of interest mask to cutoff  excess length of lines
    final_image = region_of_interest(final_image, [trapezoid])
    
    #overlay the lanes onto original image and return it
    result = weighted_img(final_image, image, α=0.6, β=1.0, λ=0.)
    
    return result


# Let's try the one with the solid white lane on the right first ...

# In[13]:

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
process_image.lines_left = []
process_image.lines_right = []
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[14]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[15]:

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[16]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

