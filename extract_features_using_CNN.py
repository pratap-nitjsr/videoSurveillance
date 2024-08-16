import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import GlobalAveragePooling2D
import pandas as pd

model = VGG16(weights='imagenet', include_top=False)

def process_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    features = GlobalAveragePooling2D()(features).numpy()
    return features

def store_features_csv(features_list, output_path):
    df = pd.DataFrame(features_list)
    df.to_csv(output_path, index=False)

image_dir = 'Output_Images_intervals'
output_path = 'CNN_features.csv'

features_list = []

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    features = process_image(image_path)
    flattened_features = features.flatten()
    features_list.append([image_name] + flattened_features.tolist())

store_features_csv(features_list, output_path)
