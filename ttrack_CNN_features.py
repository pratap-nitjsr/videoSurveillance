import os
import numpy as np
import pandas as pd
import cv2
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
from cvzone import stackImages

model = VGG16(weights='imagenet', include_top=False)

def process_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    features = GlobalAveragePooling2D()(features).numpy()
    return features

def load_saved_features(csv_path):
    df = pd.read_csv(csv_path)
    image_names = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return image_names, features

def compare_features(query_features, saved_features):
    similarities = cosine_similarity(query_features, saved_features)
    return similarities

def get_top_matches(similarities, image_names, top_n=10):
    top_indices = np.argsort(similarities, axis=1)[:, -top_n:][:, ::-1]
    top_matches = image_names[top_indices]
    return top_matches

def main(query_image_dir, saved_features_path):
    saved_image_names, saved_features = load_saved_features(saved_features_path)

    for query_image_name in os.listdir(query_image_dir):
        query_image_path = os.path.join(query_image_dir, query_image_name)
        query_features = process_image(query_image_path).flatten().reshape(1, -1)
        similarities = compare_features(query_features, saved_features)
        top_matches = get_top_matches(similarities, saved_image_names, top_n=10)

        match_img_list = []
        for match in top_matches[0]:
            match_img_list.append(cv2.imread(f'Output_Images_intervals/{match}'))

        img_stk = stackImages(match_img_list, 3, 1)
        cv2.imshow("Matches", img_stk)
        cv2.waitKey(0)


if __name__ == "__main__":
    query_image_dir = 'Query_images'
    saved_features_path = 'CNN_features.csv'
    main(query_image_dir, saved_features_path)