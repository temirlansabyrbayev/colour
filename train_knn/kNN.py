from sklearn.neighbors import KNeighborsClassifier
import csv
import pickle
from itertools import chain

def loadDataset(filename):
    training_feature_vector = []
    training_feature_lebels = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x][0:3])
            training_feature_lebels.append(dataset[x][3:4])
    training_feature_lebels = [j for sub in training_feature_lebels for j in sub]
    return training_feature_vector, training_feature_lebels

def train():

    neigh = KNeighborsClassifier(n_neighbors=10)
    X,y = loadDataset('training.data')
    neigh.fit(X, y)

    knnPickle = open('../knnpickle_file', 'wb')
    pickle.dump(neigh, knnPickle)

    knnPickle.close()

train()

