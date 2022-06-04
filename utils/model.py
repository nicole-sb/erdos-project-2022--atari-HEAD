import cv2
import numpy as np
import pandas as pd
from pyparsing import OneOrMore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def get_model_response(model, X):
    action_int = model.predict(X)[0]
    mapping = {0: "up", 1: "right", 2: "left", 3: "down,"}
    return mapping[action_int]


def train_model():

    train = pd.read_csv("Data_handling/finaldf.csv")

    X = train.drop(["action"], axis=1).copy()
    y = train["action"]

    X_train, _, y_train, _ = train_test_split(
        X, y, shuffle=True, random_state=435, test_size=0.2, stratify=y
    )

    X_train = X_train.drop(["frame_id"], axis=1).copy().values
    y_train = y_train.values

    mlp1 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=5000)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    mlp1.fit(X_train, y_train)

    return mlp1


def findang(x, y):  # calculate the angle of a unit vector
    try:
        if x < 0 and y > 0:
            theta = np.arctan(y / x) * 180 / np.pi + 180
            # theta is the angle of a vector
        elif x < 0 and y < 0:
            theta = np.arctan(y / x) * 180 / np.pi - 180
        else:
            theta = np.arctan(y / x) * 180 / np.pi
    except Exception as ex:
        theta = 0
        pass
    return theta


def findangamp(x, y):  # calculate the angle and amplitude of a vector
    amp = np.sqrt(x**2 + y**2)  # this is amplitude of a vector
    if x < 0 and y > 0:
        theta = np.arctan(y / x) * 180 / np.pi + 180
        # theta is the angle of a vector
    elif x < 0 and y < 0:
        theta = np.arctan(y / x) * 180 / np.pi - 180
    else:
        theta = np.arctan(y / x) * 180 / np.pi
    return [theta, amp]


def findobject(originalimage, pacposition):
    bglist = [
        146,
        32,
        132,
        214,
        170,
        0,
        167,
    ]  # this list contains the colorspace of the background
    # given the image array and pacposition, calculate the nearest object(ghost) near to the pacman
    image = originalimage[:172, :160]  # exclude the outside framing
    uniquelist = np.unique(image.ravel())  ##all unique gpd values in one image
    dic = (
        {}
    )  # keys will be the amplitude between pacman and ghost, values will be the list [amplitude,angle]
    for colornum in uniquelist:  # iterate all unique bgr colorspace values
        objectposition = np.nan
        if (
            colornum in bglist
        ):  # bglist contains all colorspace value of background, which is not ghost.
            continue
        else:
            result = np.where(
                image == colornum
            )  # find the object inside the image array
            if (
                np.size(result[0], axis=None) >= 5
            ):  # make sure the number of object points is large enough to satisfy the pca requirement
                lenresult = len(result[0])
                X = np.ones(
                    (lenresult, 2)
                )  # X is np array to store the points of object
                X[:, 0] = result[1]
                X[:, 1] = result[0]
                # print(X)

                pca = PCA(2)

                ## Fit the data
                pca.fit(X)
                objectposition = (
                    pca.mean_
                )  # using PCA, take mean point as the position of the object
                x = (
                    objectposition[0] - pacposition[0]
                )  # x coordinate difference between ghost and pacman
                y = (
                    objectposition[1] - pacposition[1]
                )  # y coordinate difference between ghost and pacman
                findangamp(x, y)
                dic[findangamp(x, y)[1]] = findangamp(x, y)
            else:
                objectposition = np.nan
    if dic == {}:
        return [np.nan, np.nan]
    else:
        near = min(list(dic.keys()))  # catch the nearest ghost
        return dic[near]  # [amplitude,angle]


def mostvalue(
    arr,
):  # this function is used to find the most frequent number in 2D array matrix
    arr1 = arr.flatten()
    if np.array_equal(arr1, np.array([])):
        return np.nan
    else:
        return np.bincount(arr1).argmax()


def possibleaction(
    trial, minrow, maxrow, mincol, maxcol
):  # extract the possible action by checking four blocks around pac_man in each frame
    action = []
    right = trial[minrow:maxrow, maxcol + 1 : maxcol + 9]
    left = trial[minrow:maxrow, mincol - 8 : mincol]
    up = trial[minrow - 8 : minrow, mincol:maxcol]
    down = trial[maxrow + 1 : maxrow + 9, mincol:maxcol]
    if mostvalue(up) < 60:
        action.append(0)  # 0 means up action
    if mostvalue(right) < 60:
        action.append(1)  # 1 means right action
    if mostvalue(left) < 60:
        action.append(2)  # 2 means left action
    if mostvalue(down) < 60:
        action.append(3)  # 3 means down action
    return action


def findpacman(
    image, colornum
):  # use image array and color space of pac_man to find the position of pac_man and next possible action
    result = np.where(
        image == colornum
    )  # find the pac_man, 167 is the color space of pac_man
    # print(result)
    pac = np.nan
    poss = np.nan
    if (
        np.size(result[0], axis=None) >= 5
    ):  # make sure the number of pac_man points is large enough to satisfy the pca requirement
        lenresult = len(result[0])
        X = np.ones((lenresult, 2))  # X is np array to store the points of pac_man
        X[:, 0] = result[1]
        X[:, 1] = result[0]
        # print(X)

        pca = PCA(2)

        ## Fit the data
        pca.fit(X)
        pac = pca.mean_  # using PCA, take mean point as the position of pac_man
        minrow = result[0].min()
        maxrow = result[0].max()
        mincol = result[1].min()
        maxcol = result[
            1
        ].max()  # the above four parameters represent the range of pac_man
        poss = possibleaction(image, minrow, maxrow, mincol, maxcol)
    else:
        pac = np.nan
        poss = np.nan
    return [pac, poss]


def get_photo_information(frame_id, gaze_x, gaze_y):

    path = frame_id  # not included in the repository file, and all_images file should inclue all images in the
    # trial 593_RZ_5037271_Aug-05-15-35-12
    src = cv2.imread(path)  # read image
    image = cv2.cvtColor(
        src, cv2.COLOR_BGR2GRAY
    )  # Using cv2.COLOR_BGR2GRAY color space

    pac_position = findpacman(image, 167)[0]  # 167 is the color space of pac_man
    pos_action = findpacman(image, 167)[1]

    objectlist = findobject(image, pac_position)
    ghost_amp = objectlist[1]
    ghost_angle = objectlist[0]

    gaze_mean = [gaze_x, gaze_y]

    meandiff = gaze_mean - pac_position

    mean_angle = findangamp(meandiff[0], meandiff[1])[0]
    mean_amplitude = findangamp(meandiff[0], meandiff[1])[1]

    startdiff = gaze_mean
    start_angle = findangamp(startdiff[0], startdiff[1])[0]
    start_amplitude = findangamp(startdiff[0], startdiff[1])[1]

    enddiff = gaze_mean
    end_angle = findangamp(enddiff[0], enddiff[1])[0]
    end_amplitude = findangamp(enddiff[0], enddiff[1])[1]

    gaze_variance_0 = 0
    gaze_variance_1 = 0

    comp0 = [1, 0]
    com_angle0 = findang(comp0[0], comp0[1])

    comp1 = [0, 1]
    com_angle1 = findang(comp1[0], comp1[1])

    zero = 1 if 0 in pos_action else 0
    one = 1 if 1 in pos_action else 0
    two = 1 if 2 in pos_action else 0
    three = 1 if 3 in pos_action else 0

    model_info = pd.DataFrame(
        {
            "gaze_variance_0": (gaze_variance_0),
            "gaze_variance_1": (gaze_variance_1),
            "ghost_amp": (ghost_amp),
            "ghost_angle": (ghost_angle),
            "mean_angle": (mean_angle),
            "mean_amplitude": (mean_amplitude),
            "start_angle": (start_angle),
            "start_amplitude": (start_amplitude),
            "end_angle": (end_angle),
            "end_amplitude": (end_amplitude),
            "com_angle0": (com_angle0),
            "com_angle1": (com_angle1),
            "0": (zero),
            "1": (one),
            "2": (two),
            "3": (three),
        },
        index=[0],
    )

    return model_info
