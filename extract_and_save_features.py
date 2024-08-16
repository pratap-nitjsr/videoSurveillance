import cv2
import numpy as np
import pandas as pd
import os
import warnings

warnings.simplefilter('ignore')


def calculate_mean(data):
    return np.mean(data)


def calculate_variance(data):
    return np.var(data)


def calculate_skewness(data):
    mean = calculate_mean(data)
    variance = calculate_variance(data)
    if (variance == 0):
        return 0
    n = len(data)
    skewness = np.sum((data - mean) ** 3) / (n * variance ** (3 / 2))
    return skewness


def calculate_kurtosis(data):
    mean = calculate_mean(data)
    variance = calculate_variance(data)
    if (variance == 0):
        return 0
    n = len(data)
    kurtosis = np.sum((data - mean) ** 4) / (n * variance ** 2) - 3
    return kurtosis


def split_color_channels(img):
    return (img[:, :, 0], img[:, :, 1], img[:, :, 2])


def get_histogram(img):
    hist = [0] * 256
    for i in img.flatten():
        hist[i] += 1
    return np.array(hist) / (img.shape[0] * img.shape[1])


def get_statistical_features(hist):
    feature_vector = []
    for i in range(8):
        mean = calculate_mean(hist[i * 32:(i + 1) * 32])
        variance = calculate_variance(hist[i * 32:(i + 1) * 32])
        skewness = calculate_skewness(hist[i * 32:(i + 1) * 32])
        kurtosis = calculate_kurtosis(hist[i * 32:(i + 1) * 32])
        feature_vector.extend([mean, variance, skewness, kurtosis])
    return feature_vector


def calculate_histogram(img):
    hist = [0] * 256
    for pixel in img.flatten():
        hist[pixel] += 1

    final_hist = []
    total_pixels = img.size
    cumulative_sum = 0
    threshold = total_pixels * 0.1

    for i in range(256):
        cumulative_sum += hist[i]
        while cumulative_sum >= threshold:
            final_hist.append(i)
            cumulative_sum -= threshold

    while len(final_hist) < 10:
        final_hist.append(255)

    if len(final_hist) > 10:
        final_hist = final_hist[:10]

    print("Final Hist: ", final_hist)
    return final_hist

def make_feature_vector(img):

    color_ch = split_color_channels(img)
    feat = []
    for c_img in color_ch:
        hist = get_histogram(c_img)
        feat.extend(get_statistical_features(hist))
    return feat


def save_features(folder, dest_file):
    img_list = os.listdir(folder)
    feature_vector = []
    for img_name in img_list:
        img = cv2.imread(os.path.join(folder,img_name))
        # img = cv2.resize(img, (200,200))
        feature_vector.append([img_name]+make_feature_vector(img))
    feature_vector = pd.DataFrame(np.array(feature_vector))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    feature_vector.to_csv(dest_file, index=False)


if __name__ == "__main__":
    save_features('Output_Images_intervals', 'features.csv')
