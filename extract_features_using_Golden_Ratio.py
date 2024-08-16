import cv2
import numpy as np
import os
import pandas as pd
from extract_and_save_features import make_feature_vector

fib_seq = [233, 144, 89, 55, 34, 21, 13, 8, 5]


def break_image(image):
    parts = [image, ]
    for i in range(2, 9):
        img = parts[-1]
        if img.shape[0] > img.shape[1]:
            parts.append(img[:fib_seq[i], :])
        else:
            parts.append(img[:, :fib_seq[i]])
    return parts


def get_feature(image):
    feat = []

    image_directions = [image, cv2.flip(image, 0), cv2.flip(image, 1), cv2.flip(image, -1)]

    for image_dir in image_directions:
        for part in break_image(image_dir):
            feat.extend(make_feature_vector(part))

    return np.array(feat)


def save_features(folder):
    feature_vector = []
    img_list = os.listdir(folder)
    print(len(img_list))
    for image_path in img_list:
        img = cv2.imread(f'{folder}/{image_path}')
        img = cv2.resize(img, (233, 144))
        feature_vector.append([image_path] + (get_feature(img).flatten()).tolist())
    # print (feature_vector)
    feature_vector = np.array(feature_vector)
    # feature_vector = (feature_vector - np.mean(feature_vector, axis=0))/np.var(feature_vector,axis=0)
    df = pd.DataFrame(feature_vector)
    df.to_csv('Golden_Ratio_features.csv', index=False)



if __name__ == "__main__":
    save_features('Output_Images_intervals')