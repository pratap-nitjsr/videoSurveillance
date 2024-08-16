import cv2
import numpy as np
import pandas as pd
import os
from cvzone import stackImages
from extract_and_save_features import make_feature_vector

def load_saved_features(csv_path):
    df = pd.read_csv(csv_path)
    image_names = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return image_names, features

def euclidean_distance(query_features, saved_features):
    distances = np.sqrt(np.sum((saved_features - query_features) ** 2, axis=1))
    return distances

def get_top_matches(distances, image_names, top_n=10):
    top_indices = np.argsort(distances)[:top_n]
    top_matches = image_names[top_indices]
    return top_matches

def track(query_image_dir, saved_features_path):
    saved_image_names, saved_features = load_saved_features(saved_features_path)

    for query_image_name in os.listdir(query_image_dir):
        query_img = cv2.imread(f'{query_image_dir}/{query_image_name}')
        # query_img = cv2.resize(query_img, (200,200))
        query_features = np.reshape(make_feature_vector(query_img), (1, saved_features.shape[1]))

        distances = euclidean_distance(query_features, saved_features)
        top_matches = get_top_matches(distances, saved_image_names, top_n=10)

        dataset_details = pd.read_csv('data_interval_frames.csv', index_col=False)

        print(query_image_name.upper(), ":")

        for match in top_matches:
            details = match.split('_')
            frame_no = int(details[1])
            object_no = int(details[3])

            print("-" * 50)
            print(dataset_details[(dataset_details['frame_no'] == frame_no) & (dataset_details['object_no'] == object_no)].values.astype(int))

        print("*" * 100)

if __name__ == "__main__":
    track('Query_images', 'features.csv')
