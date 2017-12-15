import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

## Use opencv to get SURF features for a set of images
def get_surf_features(images, hessian_threshold=400, upright=True, extended=False):
	surf = cv2.xfeatures2d.SURF_create(400,upright=upright,extended=extended)
	ret = [surf.detectAndCompute(img,None) for img in images]
	kps_lst = [curr[0] for curr in ret]
	des_lst = [curr[1] for curr in ret]

	return kps_lst, des_lst

def main():
	
	if len(sys.argv) != 2:
		print("Usage: python3 surf.py *image_files_directory*")
		return

	imgdir_in = sys.argv[1]
	images = []

	print("Reading images....")
	src_path = os.path.abspath(imgdir_in) + "/"
	for image_name in os.listdir(imgdir_in):
		if image_name.startswith('.'): continue
		img_path = src_path + image_name
		images.append(cv2.imread(img_path))

	kps_lst, des_lst = get_surf_features(images)

	des_lst = np.array(des_lst)
	print("des_lst (with shape {}):".format(des_lst.shape))
	print(des_lst)

	for i in range(len(images)):
		test = cv2.drawKeypoints(images[i],kps_lst[i],None,(255,0,0),4)
		print(des_lst[i])
		print("Shape of descriptor: {}".format(des_lst[i].shape))
		print("")
		plt.imshow(test),plt.show()
		plt.imshow(des_lst[i], interpolation='none')

if __name__ == "__main__":
	main()