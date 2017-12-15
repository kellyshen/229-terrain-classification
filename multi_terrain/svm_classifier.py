import cv2
from surf import get_surf_features
import numpy as np
from sklearn.externals import joblib

class SVM_Classifier():

	def __init__(self, model_directory):

		############ Load Model ###################
		self.svm = joblib.load(model_directory + '/svm.pkl')
		self.kmeans = joblib.load(model_directory + '/kmeans.pkl')
		self.SURF_params = self.get_surf_params(model_directory + '/params.txt')
		self.kmeans.verbose = False
		###########################################

	## Should return an int label. *img* should be an OpenCV Mat object
	def classify(self, img):
		
		## Get SURF features
		kps, descriptors = get_surf_features([img],
			hessian_threshold=int(self.SURF_params["hessian_threshold"]),
			upright=self.str2bool(self.SURF_params["upright"]),
			extended=self.str2bool(self.SURF_params["extended"]))

		descriptors = descriptors[0]
		if descriptors is None: return -1
		try:
			descriptors_np = np.array([np.array(curr_des) for curr_des in descriptors])
		except:
			print("Descriptors: {}".format(descriptors))
		centroid_labels = self.kmeans.predict(descriptors_np)

		## Build histogram
		vocab_size = int(self.SURF_params["vocab_size"])
		feature_vec = np.zeros((vocab_size,),dtype=int)
		for centroid in centroid_labels: feature_vec[centroid] += 1
		
		## Get label from SVM
		pred_label = self.svm.predict([feature_vec])[0]

		return pred_label

	## Load saved surf params
	def get_surf_params(self, filename):
		
		params_dict = {}
	
		param_file = open(filename,'r')
		for param in param_file:
			lst = (param.rstrip()).split(',')
			params_dict[lst[0]] = lst[1]

		return params_dict

	def str2bool(self, v):
	    if v.lower() in ('yes', 'true'):
	        return True
	    elif v.lower() in ('no', 'false'):
	        return False
	    else:
	        raise argparse.ArgumentTypeError('Boolean value expected.')