How to run the code:


First, run pip3 install -r requirements.txt to install necessary dependencies. 

Them, run the program by running python main.py. Default parameters are set for running the ICP algorithm, without merging. Full list of parameters can be found below

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