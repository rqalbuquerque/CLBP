import numpy as np
import cv2
import math
from CompletedLocalBinaryPattern import CompletedLocalBinaryPattern
import argparse
import os.path
import sys
#from tabulate import tabulate


if __name__ == "__main__":

	# args
	ap = argparse.ArgumentParser(description='Completed Local Binary Pattern (CLBP)')

	# running mode
	ap.add_argument("-m", "--mode", required=True,
		default="m1",
		help="(options: m1=Image, m2=Classification)")
	
	# database
	ap.add_argument("-t", "--training", required=False,
		help="path to the training image(s)")
	ap.add_argument("-e", "--testing", required=True, 
		help="path to the testing image(s)")

	# parameters
	ap.add_argument("-r", "--radius", required=False,
		default=1,
		help="radius from center pixel to neighbor pixels (default=1)")
	ap.add_argument("-p", "--npixels", required=False, 
		default=8,
		help="number of neighbor pixels around of center pixel (default=8)")
	# ap.add_argument("-f", "--feature", required=False, 
	# 	help=tabulate([['LBP','lbp'],
	# 		['CLBP_S_riu2','clbp_s'],
	# 		['CLBP_M_riu2','clbp_m'],
	# 		['CLBP_M_riu2/C','clbp_m/c'],
	# 		['CLBP_S_riu2_M_riu2/C','clbp_s_m/c'],
	# 		['CLBP_S_riu2/M_riu2','clbp_s/m'],
	# 		['CLBP_S_riu2_M_riu2/C','clbp_s_m/c'],
	# 		['CLBP_S_riu2/M_riu2/C','clbp_s/m/c'],
	# 		['(default)','lbp']],
	# 		headers=['Feature mapping','code']))
	ap.add_argument("-f", "--feature", required=True, 
		default="lbp",
		help="(options(mode 1): lbp, clbp_s, clbp_s_riu2, clbp_m, clbp_m_riu2, clbp_c" + 
			"(options(mode 2): lbp, clbp_s, clbp_m, clbp_m/c, clbp_s_m/c, clbp_s/m, clbp_s_m/c, clbp_s/m/c")
	args = vars(ap.parse_args())

	# verify args
	if args["mode"] == "m1":
		if os.path.exists(args["testing"])==False:
			sys.exit("File " + args["testing"] + " not found!")		
	elif args["mode"] == "m2":
		if args["training"] == None:
			sys.exit("Error! Training folder path is missing!")
		if os.path.isdir(args["training"])==False:
			sys.exit("Error! Folder " + args["training"] + " not found!")
		if os.path.isdir(args["testing"])==False:
			sys.exit("Error! Folder " + args["testing"] + " not found!")				
	else:
		sys.exit("Mode " + args["mode"] + " is not valid!")		

	# run
	if args["mode"] == "m1":

		file = args["testing"]
		img = cv2.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		radius = int(args["radius"])
		numPixels = int(args["npixels"])
		mapping = args["feature"]
		
		clbp = CompletedLocalBinaryPattern(radius,numPixels,mapping)
		lpImg = np.array(clbp.genLocalPatterns(gray),dtype=np.uint8)

		# display the local patterns
		cv2.imshow("Image", gray)
		cv2.imshow("Local patterns", lpImg)
		cv2.waitKey(0)

