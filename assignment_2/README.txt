This is assignment 1 for Computer Vision 2. We have chosen to implement this assignment in python 3.7, so this is a requirement. Our code can be found in the Code folder.

How to run the code:

First, run pip install -r requirements.txt to install necessary dependencies. 

Then, make sure you are located in the code/ directory. 


Fundamental Matrix (Section 3)

In order to run the various implementation for estimating the fundamental matrix and drawing the epipolar lines, run the following program. 

python epa.py

Default parameters are set for running the algorithm with. Full list of parameters can be found below. 

-h, --help    	--> show this help message and exit
--image_1 	--> fist image
--image_2 	--> second image
--f_method 	--> {normal,ransac,opencv}   method for constructing fundamental matrix
--normalize 	--> sse normalized points
--ransac_iters	--> number of ransac iterations
--sampson_threshold --> sampson_threshold for ransac distances
--dist_filter 	--> initial points filtering



Chaining (Section 4)

In order to run the program to create the point-view matrix, run 

python Code/chaining.py

Default parameters are set to create the matrix with Brute Force matching, using 50% of all matches. The full list of parameters is as follows:

-h, --help   	--> show this help message and exit
--viz     	--> whether to visualise the result
--match_method	--> {bf,flann} which method to use for matching feature points (brute force or Flann)
--dist_filter 	--> initial points filtering
--nearby_filter	--> threshold for determining whether two points are similar 



Structure from Motion (Section 5)

NOTE: python chaining.py MUST be run before running this section, as the point-view matrix needs to be generated beforehand! 

In order to run the program to create the 3D reconstruction run

python structure_from_motion.py  

The corresponding parameters are:

-h, --help  	--> 	show this help message and exit
--visualize 	-->    	whether to visualize the result
--match_method  --> 	{bf,flann} which method to use for matching feature points
--dist_filter  	-->	initial points filtering
--nearby_filter	-->	threshold for determining whether two points are
                        similar
--consecutive_frames --> amount of consecutive frames
--stitching_method  -->	{icp,pr} which method to use for stitching

