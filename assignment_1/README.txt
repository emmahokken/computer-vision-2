This is assignment 1 for Computer Vision 2. We have chosen to implement this assignment in python 3.7, so this is a requirement. 

How to run the code:


First, run pip install -r requirements.txt to install necessary dependencies. 

Them, run the program by running python main.py. Default parameters are set for running the ICP algorithm, without merging. Full list of parameters can be found below. 

-h, --help	-->   	show this help message and exit
--merge        	-->	whether to merge the pcds
--start 	-->   	first pcd
--end 		-->   	final pcd
--step_size 	--> 	step size between the pcds
--sampling_method --> 	{uniform, random, normal} method for sub sampling pcd rows
--sampling_r 	--> 	ratio for sub sampling
--max_icp_iters	--> 	max number of iterations for icp algorithm
--icp_treshold 	--> 	threshold for early stopping icp algorithm
--icp_treshold_w --> 	window for treshold for icp algorithm
--noise_treshold --> 	keep points up to this distance
--visualize 	--> 	whether to visualize the result
--merge_method	-->	{3.1,3.2} method for merging, always use uniform sampling
--dist_measure	-->	{nn,kd} method for measuring distance, nearest neighbour or kd-trees


If one wishes to merge images the code should be run as follows.

python Code/main.py --merge True. 


If one wishes to only run the ice algorithm on for example the first two images, the code should be run as follows 

python Code/main.py --start 0 --end 1
