import numpy as np
import os
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from feature_extraction import get_train_test
from config import args


def get_XY(dirpath, encoder_kind: str = None, scaling_kind: str = None, force_update=False, debug=False):

    encoder = None
    encoders={
        "ONEHOT": OneHotEncoder(sparse=False, handle_unknown='ignore'),
        "LABEL": LabelEncoder()
    }

    scalers={
        "STANDARDIZE": StandardScaler(),
        "NORMALIZE": MinMaxScaler(feature_range=(0, 1))
    }

    x_train_filepath = os.path.join(dirpath, "x_train.npy")
    y_train_filepath = os.path.join(dirpath, "y_train.npy")
    x_test_filepath = os.path.join(dirpath, "x_test.npy")
    y_test_filepath = os.path.join(dirpath, "y_test.npy")

    if ((not os.path.exists(x_train_filepath)) or
            (not os.path.exists(y_train_filepath)) or
            (not os.path.exists(x_test_filepath)) or
            (not os.path.exists(y_test_filepath)) or
            force_update):

        if debug:
            print("Creating XY")

        train_df, test_df = get_train_test(dirpath, force_update, debug)

        x_train, y_train = train_df.iloc[:, 1:-1].values, train_df.iloc[:, -1].values
        x_test, y_test = test_df.iloc[:, 1:-1].values, test_df.iloc[:, -1].values

        if encoder_kind.upper() == "ONEHOT":
            y_train = y_train.reshape([-1, 1])
            y_test = y_test.reshape([-1, 1])

        if scaling_kind is not None:
            scaler = scalers[scaling_kind.upper()]
            x_train[:,1:] = scaler.fit_transform(x_train[:,1:])
            x_test[:, 1:] = scaler.transform(x_test[:, 1:])

        if debug:
            print("Saving XY")

        np.save(x_train_filepath, x_train)
        np.save(y_train_filepath, y_train)
        np.save(x_test_filepath, x_test)
        np.save(y_test_filepath, y_test)

    else:
        if debug:
            print("Reading XY")

        x_train = np.load(x_train_filepath, allow_pickle=True)
        y_train = np.load(y_train_filepath, allow_pickle=True)
        x_test = np.load(x_test_filepath, allow_pickle=True)
        y_test = np.load(y_test_filepath, allow_pickle=True)

    if encoder_kind is not None:
        encoder = encoders[encoder_kind.upper()]
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

    return (x_train, y_train), (x_test, y_test), encoder


def train_SVM(train_data,test_data, debug=False):

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    print('--------------------------------------------------SVM----------------------------------------------------')

    C_parm = np.logspace(0, 10, 13)
    G_parm = np.logspace(-9, 3, 13)
    kernel = ['rbf']
    Accuracies = []

    C_sel = 0
    G_sel = 0
    K_sel = " "
    Acc_sel = 0

    # Select best C and gamma values
    for k in kernel:
        for C in C_parm:
            for g in G_parm:
                clf = svm.SVC(kernel=k, C=C, gamma=g)
                clf = clf.fit(x_train, y_train)
                prediction = clf.predict(x_test)
                score = accuracy_score(y_test, prediction) * 100
                Accuracies.append(score)

                print("Kernel = " + str(k) + " C = " + str(C) + " Gamma = " + str(g) + ", Accuracy = " + str(score))
                if (score > Acc_sel):
                    Acc_sel = score
                    C_sel = C
                    G_sel = g
                    K_sel = k

    print("Best parameter values are: C= " + str(C_sel) + " Gamma=" + str(G_sel) + " Kernel=" + str(K_sel))


def train_KNN(train_data,test_data,encoder,debug=False):

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    print('--------------------------------------------------KNN----------------------------------------------------')
    neighs = range(31,45,2)
    weighs = {'distance'}
    acc_sel = 0

    # Select best parameters
    for neigh in neighs:
        for weigh in weighs:
            clf = neighbors.KNeighborsClassifier(n_neighbors=neigh, weights=weigh, algorithm='auto')
            clf = clf.fit(x_train, y_train)
            prediction = clf.predict(x_test)
            score = accuracy_score(y_test, prediction) * 100
            print(
                "Neighbours = " + str(neigh) + ", Weighting : " + weigh + ", Accuracy = " + str(
                    score))
            if (score > acc_sel):
                acc_sel = score
                n_sel = neigh
                w_sel = weigh


    print("Best values are: Neighbours = " + str(n_sel) + ", Weighting : " + w_sel + ", Accuracy = " + str(acc_sel))


train_data, test_data, encoder = get_XY(args.dataset, encoder_kind="LABEL", scaling_kind="STANDARDIZE", debug=True)
train_SVM(train_data, test_data, encoder)