import numpy as np
import os
from scipy import io
import matplotlib.pyplot as plt
import itertools

def load_off(filename, size):
    """ Read Object File Format (.off) into Numpy 3D array. """

    # create 3D array (cube with edge = size)
    obj = np.zeros([size, size, size])

    # open filename.off
    with open(filename) as f:

        # read first line
        header = f.readline()                   # returns a string
        # set properties
        properties = f.readline().split(" ")    # returns a list of chars
        num_vertices = int(properties[0])
        num_faces = int(properties[1])
        num_edges = int(properties[2])
        print("Properties:",
              "\nNumber of vertices:", num_vertices,
              "\nNUmber of faces:   ", num_faces,
              "\nNumber of edges:   ", num_edges)

        # read everything else
        body = f.readlines()                    # returns a list of strings
        if num_vertices != 0:
            vertices = body[0:num_vertices]
        else:
            raise ValueError("No vertex found.")
        if num_faces != 0:
            faces = body[num_vertices:num_vertices+num_faces]
        else:
            raise ValueError("No face found.")
        if num_edges != 0:
            edges = body[num_faces:num_faces+num_edges]
        
        # set vertices
        for i in range(num_vertices):
            coords = vertices[i].split(" ")
            if (int(float(coords[0])) < size) and (int(float(coords[1])) < size) and (int(float(coords[2])) < size):
                obj[int(float(coords[0])), int(float(coords[1])), int(float(coords[2]))] = 1
            else:
                print("Error at vertex", i)

        return obj

def load_folder(folder, size):
    """ Load all files from a folder into a 4D nupmy array. """

    # create a 4D array with first dimension the number of files
    num_files = len(os.listdir(folder))
    print(folder, "contains", num_files, "objects.")
    dataset = np.zeros([num_files, size, size, size])

    for index, filename in enumerate(os.listdir(folder)):
        print("\nImporting:", filename)
        dataset[index, :, :, :] = load_off(folder + filename, size)

    return dataset

def load_mat(folder, train):

    # create a 4D array with first dimension the number of files
    num_files = len(os.listdir(folder))
    if train:
        print("Training set:", num_files)
    else:
        print("Test set    :", num_files)
    dataset = np.zeros([num_files, 30, 30, 30])

    for index, filename in enumerate(os.listdir(folder)):
        #print("\nImporting:", filename)
        dataset[index, :, :, :] = io.loadmat(folder + filename)['instance']

    return dataset, num_files

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')