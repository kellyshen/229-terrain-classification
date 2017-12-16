import cv2
import sys
import os
import numpy as np

PADDING = 42

def augment_data(images,
	flip=False,
	rotation=False,
	alpha_beta=False,
	gaussian_noise=False,
	# shear=False,
	# rescale=False,
	# stretch=False,
	# translation=False
	shuffle=False,
	compound=False
	):
	
	print("---------------------------")
	if alpha_beta:
		print("Adjusting alpha beta....")
		adjusted = []
	
		for img in images:
			for i in range(3):
				alpha,beta = 0.75 + np.random.rand() * 1.0, -60.0 + np.random.rand() * 100
				adjusted.append(alpha * img + beta)

		if compound: images += adjusted

	if rotation:
		print("Getting rotations....")
		rows,cols = images[0].shape[0], images[0].shape[1]
		padded_rows, padded_cols = rows+2*PADDING, cols+2*PADDING
		rotation_center = (padded_rows/2, padded_cols/2)
		rotated = []

		for img in images:
			padded = cv2.copyMakeBorder(img,PADDING,PADDING,PADDING,PADDING,cv2.BORDER_WRAP)
			M1 = cv2.getRotationMatrix2D(rotation_center,np.random.randint(15,75),1)
			M2 = cv2.getRotationMatrix2D(rotation_center,-1 * np.random.randint(15,75),1)
			M3 = cv2.getRotationMatrix2D(rotation_center,np.random.randint(0,np.random.randint(75,360)),1)
			rotated.append(cv2.warpAffine(padded,M1,(padded_rows, padded_cols))[PADDING:PADDING+rows,PADDING:PADDING+cols,:])
			rotated.append(cv2.warpAffine(padded,M2,(padded_rows, padded_cols))[PADDING:PADDING+rows,PADDING:PADDING+cols,:])
			rotated.append(cv2.warpAffine(padded,M3,(padded_rows, padded_cols))[PADDING:PADDING+rows,PADDING:PADDING+cols,:])
		
		if compound: images += rotated

	if flip:
		print("Getting flips....")
		flip0 = [ cv2.flip(img,0) for img in images ]
		flip1 = [ cv2.flip(img,1) for img in images ]
		
		if compound:
			images += flip0
			images += flip1

	if gaussian_noise:
		print("Getting noise....")
		mu,sigma = np.array([0,0,0]), np.array([20,20,20])
		noisy = [ img + cv2.randn(img.copy(),mu,sigma) for img in images ]
		if compound: images += noisy

	if not compound:
		if alpha_beta: images += adjusted
		if rotation: images += rotated
		if flip: images += flip0 + flip1
		if gaussian_noise: images += noisy

	if shuffle:
		print("Shuffling")
		np.random.shuffle(images)
	print("---------------------------")
		
def main():
	if len(sys.argv) != 3:
		print("ERROR! Usage: python3 augment.py *source_directory* *output_directory*")
		return

	imgdir_in = sys.argv[1]
	imgdir_out = sys.argv[2]
	images = []

	print("Reading images....")
	src_path = os.path.abspath(imgdir_in) + "/"
	for image_name in os.listdir(imgdir_in):
		if image_name.startswith('.'): continue
		img_path = src_path + image_name
		images.append(cv2.imread(img_path))

	print("Augmenting images...")
	augment_data(images,flip=True,rotation=True,alpha_beta=True,gaussian_noise=True,shuffle=True)

	print("Writing images...")
	out_path = os.path.abspath(imgdir_out) + "/"
	index = 0

	num_images = len(images)
	for image in images:
		if index%5==0: print("\r{}% complete  ".format(int(float(index)/float(num_images)*100)),end='')
		cv2.imwrite(out_path + str(index) + ".png", image)
		index += 1
	print("")

if __name__ == "__main__":
	main()