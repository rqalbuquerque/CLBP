import numpy as np
import cv2
from CompletedLocalBinaryPattern import CompletedLocalBinaryPattern
from sklearn.neighbors import KNeighborsClassifier

def chiSquared(p,q):
    return np.sum((p-q)**2/(p+q+1e-6))


if __name__ == "__main__":

	database = "C:/Users/rqa/Desktop/CLBP/database/Outex-TC-00010/000/"

	radius = 1
	numPixels = 8
	mapping = "lbp"
	clbp = CompletedLocalBinaryPattern(radius,numPixels,mapping)

	data = []
	labels = []

	model = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)


	# classes
	with open(database + "classes.txt", "r") as file:
		numClasses = int(file.readline())
		classes = {}
		for line in file:
			columns = line.split()
			classes[columns[0]] = int(columns[1])

	# train
	with open(database + "train_mini.txt", "r") as train:
		with open(database + "train_files.txt", "r") as trainFiles:
			numTrain = int(train.readline())
			i = 0
			for pathFile in trainFiles:
				print(pathFile)
				img = cv2.imread(pathFile.rstrip(), cv2.IMREAD_GRAYSCALE)
				img = (img-np.mean(img))/np.std(img)*20+128
				hist = clbp.describe(img)
				labels.append(train.readline().split()[1])
				data.append(hist)

				i += 1
				if i>=numTrain:
					break


