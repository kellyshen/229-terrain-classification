import sys
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from surf import get_surf_features
import numpy as np
from sklearn.externals import joblib
import datetime
import dateutil.tz

################## Data ###################

train_data = [
"./_split/train_aug/bark",
"./_split/train_aug/dirt",
"./_split/train_aug/dry_veg",
"./_split/train_aug/foliage",
"./_split/train_aug/grass",
"./_split/train_aug/paved"
]

val_data = [
"./_split/validation/bark",
"./_split/validation/dirt",
"./_split/validation/dry_veg",
"./_split/validation/foliage",
"./_split/validation/grass",
"./_split/validation/paved"
]

test_data = [
"./_split/test/bark",
"./_split/test/dirt",
"./_split/test/dry_veg",
"./_split/test/foliage",
"./_split/test/grass",
"./_split/test/paved"
]

class_labels_raw = range(len(train_data))
class_names = ['bark','dirt','dry_veg','foliage','grass','paved']

################ PARAMETERS #################

TEMP_MAX = 26000
VALIDATING = False
TESTING = True
TRAINING = not VALIDATING and not TESTING

## SURF Parameters
hessian_threshold=300
upright=True
extended=False

## K-means Parameters
vocab_size=125
precompute_distances=False
kmeans_verbose=True
n_jobs=1 # num CPUs, -1 -> all CPUs
algorithm='auto'

## SVM Parameters
svm_verbose=True
shrinking=False

def str2bool(v):
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

## Get params stored in file
def get_surf_params(filename):
	params_dict = {}
	param_file = open(filename,'r')

	for param in param_file:
		lst = (param.rstrip()).split(',')
		params_dict[lst[0]] = lst[1]

	return params_dict

def print_params(C, kernel, dict=None ):
	print("")
	print("------------------------------")
	print("General Params:")
	print("temp_max: {}".format(TEMP_MAX))
	print("")
	
	print("SURF Params:")
	print("hessian_threshold: {}".format(hessian_threshold))
	print("")

	print("KMeans (minibatch) Params:")
	print("vocab_size: {}".format(vocab_size))
	print("")
				
	print("SVM params:")
	print("C: {}".format(C))
	print("kernel: {}".format(kernel))
	print("shrinking: {}".format(shrinking))
	print("------------------------------")
	print("")

def save_params(filename,C,kernel):
	with open(filename,'w') as f:
		f.write("hessian_threshold,{}\n".format(hessian_threshold))
		f.write("upright,{}\n".format(upright))
		f.write("extended,{}\n".format(extended))
		f.write("vocab_size,{}\n".format(vocab_size))
		f.write("shrinking,{}\n".format(shrinking))
		f.write("C,{}\n".format(C))
		f.write("kernel,{}\n".format(kernel))

def get_data(images, folders, class_labels):
	labels = []
	for folder, class_label in zip(folders, class_labels):
		print("Currently in {}".format(folder))
		src_path = os.path.abspath(folder) + "/"

		print("Reading images....")
		
		TMP_MAX = TEMP_MAX
		CURR = 0

		for image_name in os.listdir(folder):
			
			CURR += 1
			if CURR > TMP_MAX: break

			if image_name.startswith('.'): continue
			img_path = src_path + image_name
			images.append(cv2.imread(img_path))
			labels.append(class_label)
		
		print("")
	return np.array(labels)

def main():

	train_images,val_images = [],[]
	train_folders = train_data
	val_folders = test_data if TESTING else val_data

	if not TRAINING:
		
		if len(sys.argv) != 2:
			print("ERROR! Usage: python3 train.py *load_directory (if validating or testing)*")
			return

		load_dir = './log/train/' + sys.argv[1]

		svm = joblib.load(load_dir + '/svm.pkl')
		kmeans = joblib.load(load_dir + '/kmeans.pkl')
		SURF_params = get_surf_params(load_dir + '/params.txt')

	if TRAINING:
		print("Getting train data...")
		train_labels_init = get_data(train_images,train_folders,class_labels_raw)
	else:
		print("Getting val/test data...")
		val_labels_init = get_data(val_images,val_folders,class_labels_raw)

	print("Getting SURF (train)...")
	if TRAINING:
		train_kps, train_descriptors = get_surf_features(train_images,
			hessian_threshold=hessian_threshold,
			upright=upright,
			extended=extended)
	else:
		print("Getting SURF (val/test)...")
		
		# print("hessian_threshold?: {}".format(int(SURF_params["hessian_threshold"])))
		# print("upright?: {}".format(str2bool(SURF_params["upright"])))
		# print("extended?: {}".format(str2bool(SURF_params["extended"])))
	
		val_kps, val_descriptors = get_surf_features(val_images,
			hessian_threshold=int(SURF_params["hessian_threshold"]),
			upright=str2bool(SURF_params["upright"]),
			extended=str2bool(SURF_params["extended"]))

	print("Rearranging data....")
	if TRAINING: train_descriptors_np_init = [np.array(curr_des) for curr_des in train_descriptors]
	else: val_descriptors_np_init = [np.array(curr_des) for curr_des in val_descriptors]

	train_descriptors_np, val_descriptors_np = [], []
	train_labels, val_labels = [], []

	if TRAINING:
		print("Clearing out invalid train")
		for i in range(len(train_descriptors_np_init)):
			if len(train_descriptors_np_init[i].shape) == 2: 
				train_descriptors_np.append(train_descriptors_np_init[i])
				train_labels.append(train_labels_init[i])
	else:
		print("Clearing out invalid val/test")
		for i in range(len(val_descriptors_np_init)):
			if len(val_descriptors_np_init[i].shape) == 2: 
				val_descriptors_np.append(val_descriptors_np_init[i])
				val_labels.append(val_labels_init[i])

	if TRAINING:
		print("Stacking train")
		train_descriptor_matrix = np.vstack(train_descriptors_np)
		train_image_index = np.ones((train_descriptor_matrix.shape[0],),dtype=int) * -1
	else:
		print("Stacking val/test")
		val_descriptor_matrix = np.vstack(val_descriptors_np)
		val_image_index = np.ones((val_descriptor_matrix.shape[0],),dtype=int) * -1
	
	if TRAINING:
		print("Indexer train")
		curr_pos = 0
		diff = 0
		for i in range(len(train_images)):
			if train_descriptors[i] is None: 
				diff += 1
				continue
			next_pos = curr_pos + len(train_descriptors[i])
			train_image_index[curr_pos:next_pos] = i - diff
			curr_pos = next_pos
	else:
		print("Indexer val/test")
		curr_pos = 0
		diff = 0
		for i in range(len(val_images)):
			if val_descriptors[i] is None:
				diff += 1
				continue
			next_pos = curr_pos + len(val_descriptors[i])
			val_image_index[curr_pos:next_pos] = i - diff
			curr_pos = next_pos

	if TRAINING:
		print("Running k-means")
		kmeans = MiniBatchKMeans(n_clusters=vocab_size,
			precompute_distances=precompute_distances,
			verbose=kmeans_verbose,
			# n_jobs=n_jobs, # num CPUs, -1 -> all CPUs
			# algorithm=algorithm
		).fit(train_descriptor_matrix)

		train_centroid_labels = kmeans.labels_
		centroids = kmeans.cluster_centers_
	else:
		print("Getting centroid labels (val/test)")
		val_centroid_labels = kmeans.predict(val_descriptor_matrix)

	print("Building image histograms")
	if TRAINING:
		train_features = np.zeros((len(train_labels),vocab_size),dtype=int)
		for i in range(len(train_centroid_labels)):
			img_ind = train_image_index[i]
			train_features[img_ind,train_centroid_labels[i]] += 1
	else:
		val_features = np.zeros((len(val_labels),int(SURF_params['vocab_size'])),dtype=int)
		for i in range(len(val_centroid_labels)):
			img_ind = val_image_index[i]
			val_features[img_ind,val_centroid_labels[i]] += 1

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M')

	if TRAINING:
		SVM_params = [(0.01,"linear"),(0.1,"linear"),(0.5,"linear"),(1.0,"linear"),
		(0.01,"rbf"),(0.1,"rbf"),(0.5,"rbf"),(1.0,"rbf")]

		for param_set in SVM_params:

			C,kernel = param_set

			print("Training SVM")
			svm = SVC(C=C, kernel=kernel, verbose=svm_verbose, shrinking=shrinking)
			svm.fit(train_features, train_labels)

			pred_train = svm.predict(train_features)
			train_diffs = pred_train - train_labels 
			train_error = len(np.nonzero(train_diffs)[0]) / (train_diffs.shape[0])

			print("Training error is: {}".format(train_error))

			print("Saving confusion matrix (train)...")
			confMat_train = confusion_matrix(train_labels, pred_train)
			print(confMat_train)
			plt.matshow(confMat_train)
			plt.show()

			print_params(C,kernel)

			print("Logging data to:")
			cwd = os.getcwd()
			directory = cwd + "/log/train/" + "reg_" + str(param_set[0]) + "_kernel_" + param_set[1] + timestamp
			print(directory)
			
			svm_filename = directory + '/svm.pkl'
			kmeans_filename = directory + '/kmeans.pkl'
			params_filename = directory + '/params.txt'
			train_confmat_file = directory + '/confmat_train.npy'
			
			os.makedirs(directory)

			joblib.dump(svm, svm_filename)
			joblib.dump(kmeans, kmeans_filename)
			np.save(train_confmat_file, confMat_train)
			save_params(params_filename,C,kernel)
	else:

		pred_val = svm.predict(val_features)
		val_diffs = pred_val - val_labels 
		val_error = len(np.nonzero(val_diffs)[0]) / (val_diffs.shape[0])

		print("Val error is: {}".format(val_error))

		print("Saving confusion matrix (val)...")
		confMat_val = confusion_matrix(val_labels, pred_val)
		print(confMat_val)
		plt.matshow(confMat_val)
		plt.show()

		cwd = os.getcwd()
		curr = 'val' if VALIDATING else 'test'
		directory = cwd + '/log/' + curr + '/' + timestamp
		os.makedirs(directory)
		val_confmat_file = directory + '/confmat.npy'
		np.save(val_confmat_file, confMat_val)


if __name__ == "__main__":
	main()