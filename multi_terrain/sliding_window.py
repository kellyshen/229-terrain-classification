import cv2
import numpy as np
from svm_classifier import SVM_Classifier
import time

############## Setup for data, model, and class labels ############
model_directory = #path to directory containing model files#
image_path = #path to landscape image#
save_path = #path for image to be created/saved#

class2label = {}
label2class = ['bark','dirt','dry_veg','foliage','grass','paved']
for i in range(len(label2class)): class2label[label2class[i]] = i
####################################################################

###############  Sliding Window Parameters #########################
STEP_SIZE = 20
#####################################################################

def main():

	## Define colors (openCV uses BGR)
	colors = [
		[255,0,0], # BlUE / BARK
		[0,0,255], # RED / DIRT
		[0,255,255], # YELLOW / DRY_VEG
		[0,255,0], # GREEN / FOLIAGE
		[255,0,255], # MAGENTA / GRASS
		[255,255,0] # CYAN / PAVED
	]

	print("Getting image data")
	img = cv2.imread(image_path)
	print("Img shape: {}".format(img.shape))
	dimensions = img.shape[:2]
	num_channels = img.shape[2]
	
	print("Setting up sliding window")
	patch_classifier = SVM_Classifier(model_directory)
	num_classes = len(label2class)
	counts = np.zeros((dimensions[0],dimensions[1],num_classes),dtype=int)
	curr_pt = [0,0]
	step = STEP_SIZE

	print("Performing sliding window")
	while curr_pt[1] + 99 < dimensions[1]:
		while curr_pt[0] + 99 < dimensions[0]:
			print("\rCurrent point is: {}   ".format(curr_pt),end="")
			curr_cell = img[curr_pt[0]:curr_pt[0]+100,curr_pt[1]:curr_pt[1]+100,:]
			curr_label = patch_classifier.classify(curr_cell)
			if curr_label >= 0: counts[curr_pt[0]:curr_pt[0]+100,curr_pt[1]:curr_pt[1]+100,curr_label] += 1
			curr_pt[0] += step
		curr_pt[0] = 0
		curr_pt[1] += step
	print("")

	print("Getting color labeling for image")
	pix_labels = np.argmax(counts,axis=2)
	overlay = np.zeros((dimensions[0],dimensions[1],num_channels),dtype='uint8')
	for i in range(dimensions[0]):
		for j in range(dimensions[1]):
			overlay[i,j,:] = colors[pix_labels[i,j]]
	
	print("Overlaying color labeling")
	alpha = 0.25
	ret = cv2.addWeighted(overlay[:dimensions[0]+1,:dimensions[1]+1,:],alpha,img,1-alpha,0)

	## Save image
	cv2.imwrite(save_path,ret)

	## Display output
	cv2.imshow('image',ret)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()