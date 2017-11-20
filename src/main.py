import numpy as np
import cv2
from CompletedLocalBinaryPattern import CompletedLocalBinaryPattern
#from sklearn.neighbors import KNeighborsClassifier

def chiSquared(p,q):
    return np.sum((p-q)**2/(p+q+1e-6))


if __name__ == "__main__":

	database = "C:/Users/rqa/Desktop/CLBP/database/Outex-TC-00010/000/"

	radius = 1
	numPixels = 8

	clbp_s = []
	clbp_m = []
	clbp_mc = []
	clbp_s_mc = []
	clbp_sm = []
	clbp_smc = []

	labels = []
	
	result_labels_clbp_s = []
	result_labels_clbp_m = []
	result_labels_clbp_mc = []
	result_labels_clbp_s_mc = []
	result_labels_clbp_sm = []
	result_labels_clbp_smc = []

	# nn_classifier_clbp_s = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# nn_classifier_clbp_m = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# nn_classifier_clbp_mc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# nn_classifier_clbp_s_mc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# nn_classifier_clbp_sm = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# nn_classifier_clbp_smc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)

	# classes
	with open(database + "classes.txt", "r") as file:
		numClasses = int(file.readline())
		classes = {}
		for line in file:
			columns = line.split()
			classes[int(columns[1])] = columns[0]

	# train
	print("Training steps:")
	with open(database + "train_mini.txt", "r") as train:
		with open(database + "train_files.txt", "r") as trainFiles:
			numTrain = int(train.readline())
			i = 0
			for pathFile in trainFiles:
				img = cv2.imread(pathFile.rstrip(), cv2.IMREAD_GRAYSCALE)
				img = (img-np.mean(img))/np.std(img)*20+128

				dp = calcLocalDifferences(img,numPoints,radius)
				sp, mp = LDSMT(dp)
				# clbp_s.append(describeHistogram(img, sp, mp, "clbp_s"))
   				clbp_m.append(describeHistogram(img, sp, mp, "clbp_m"))
   				# clbp_mc.append(describeHistogram(img, sp, mp, "clbp_m/c"))
   				# clbp_s_mc.append(describeHistogram(img, sp, mp, "clbp_s_m/c"))
   				# clbp_sm.append(describeHistogram(img, sp, mp, "clbp_s/m"))
   				# clbp_smc.append(describeHistogram(img, sp, mp, "clbp_s/m/c"))

				labels.append(train.readline().split()[1])

				print(i)
				i += 1
				if i>=numTrain:
					break

	# fitting
	print("Fitting the models")

	# model = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
	# model.fit(data,labels)

	# nn_classifier_clbp_s.fit(clbp_s,labels)
	# nn_classifier_clbp_m.fit(clbp_m,labels)
	# nn_classifier_clbp_mc.fit(clbp_mc,labels)
	# nn_classifier_clbp_s_mc.fit(clbp_s_mc,labels)
	# nn_classifier_clbp_sm.fit(clbp_sm,labels)
	# nn_classifier_clbp_smc.fit(clbp_smc,labels)

	# test
	print("Test steps:")
	with open(database + "test_mini.txt", "r") as test:
		with open(database + "test_files.txt", "r") as testFiles:
			numTest = int(test.readline())
			i = 0
			for pathFile in testFiles:
				img = cv2.imread(pathFile.rstrip(), cv2.IMREAD_GRAYSCALE)
				img = (img-np.mean(img))/np.std(img)*20+128

				dp = calcLocalDifferences(img,numPoints,radius)
				sp, mp = LDSMT(dp)
				hist_clbp_s = describeHistogram(img, sp, mp, "clbp_s")
   				# hist_clbp_m = describeHistogram(img, sp, mp, "clbp_m")
   				# hist_clbp_mc = describeHistogram(img, sp, mp, "clbp_m/c")
   				# hist_clbp_s_mc = describeHistogram(img, sp, mp, "clbp_s_m/c")
   				# hist_clbp_sm = describeHistogram(img, sp, mp, "clbp_s/m")
   				# hist_clbp_smc = describeHistogram(img, sp, mp, "clbp_s/m/c")

				result_labels_clbp_s.append((test.readline().split()[1],
					nn_classifier_clbp_s.predict(hist_clbp_s)))

				# result_labels_clbp_m.append((test.readline().split()[1],
				# 	nn_classifier_clbp_m.predict(hist_clbp_m)))

				# result_labels_clbp_mc.append((test.readline().split()[1],
				# 	nn_classifier_clbp_mc.predict(hist_clbp_mc)))

				# result_labels_clbp_s_mc.append((test.readline().split()[1],
				# 	nn_classifier_clbp_s_mc.predict(hist_clbp_s_mc)))

				# result_labels_clbp_sm.append((test.readline().split()[1],
				# 	nn_classifier_clbp_sm.predict(hist_clbp_sm)))

				# result_labels_clbp_smc.append((test.readline().split()[1],
				# 	nn_classifier_clbp_smc.predict(hist_clbp_smc)))

				print(i)
				i += 1
				if i>=numTrain:
					break

	result_labels_clbp_s = np.array(result_labels_clbp_s)
	acc_labels_clbp_s = (result_labels_clbp_s[:,0] == result_labels_clbp_s[:,1])/numTest

	# result_labels_clbp_m = np.array(result_labels_clbp_m)
	# acc_labels_clbp_m = (result_labels_clbp_m[:,0] == result_labels_clbp_m[:,1])/numTest

	# result_labels_clbp_mc = np.array(result_labels_clbp_mc)
	# acc_labels_clbp_mc = (result_labels_clbp_mc[:,0] == result_labels_clbp_mc[:,1])/numTest

	# result_labels_clbp_s_mc = np.array(result_labels_clbp_s_mc)
	# acc_labels_clbp_s_mc = (result_labels_clbp_s_mc[:,0] == result_labels_clbp_s_mc[:,1])/numTest

	# result_labels_clbp_sm = np.array(result_labels_clbp_sm)
	# acc_labels_clbp_sm = (result_labels_clbp_sm[:,0] == result_labels_clbp_sm[:,1])/numTest

	# result_labels_clbp_smc = np.array(result_labels_clbp_smc)
	# acc_labels_clbp_smc = (result_labels_clbp_smc[:,0] == result_labels_clbp_smc[:,1])/numTest

