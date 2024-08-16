import cv2
import numpy as np
import pandas as pd
import os
from cvzone import stackImages
from extract_features_using_Golden_Ratio import get_feature, break_image


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

def draw_bounding_box_and_save_video(match_details, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Correctly getting the total number of frames

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    for match in match_details:
        frame_no, _, _, center_x, center_y, width, height = match

        start_frame = max(0, frame_no - 150)
        end_frame = min(frame_no + 150, total_frames - 1)  # Correct usage of total frame count

        current_frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set video to start at the correct frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame == frame_no:
                top_left_x = int(center_x - width // 2)
                top_left_y = int(center_y - height // 2)
                cv2.rectangle(frame, (top_left_x, top_left_y),
                              (top_left_x + width, top_left_y + height),
                              (0, 255, 0), 2)
                cv2.imshow("Output", frame)
                cv2.waitKey(3000)

            else:
                cv2.imshow("Output", frame)
                cv2.waitKey(1)

            out.write(frame)

            current_frame += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def main(query_image_dir, saved_features_path, video_path, output_video_dir):
    saved_image_names, saved_features = load_saved_features(saved_features_path)


    for query_image_name in os.listdir(query_image_dir):
        print(query_image_name)
        query_img = cv2.imread(f'{query_image_dir}/{query_image_name}')
        query_img = cv2.resize(query_img, (233, 144))
        query_features = np.reshape(get_feature(query_img), (1, saved_features.shape[1]))

        distances = euclidean_distance(query_features, saved_features)
        top_matches = get_top_matches(distances, saved_image_names, top_n=10)

        # print(query_image_name.upper(), ":")

        dataset_details = pd.read_csv('data_interval_frames.csv', index_col=False)

        match_details = []
        # print(top_matches)
        for match in top_matches:
            details = match.split('_')
            print(details)
            frame_no = int(details[1])
            object_no = int(details[3])
            match_detail = dataset_details[(dataset_details['frame_no'] == frame_no) &
                                           (dataset_details['object_no'] == object_no)].values.astype(int)
            if len(match_detail) > 0:
                match_details.append(match_detail[0])
            break

        if len(match_details) > 0:
            output_path = os.path.join(output_video_dir, f'{query_image_name}_output.avi')
            draw_bounding_box_and_save_video(match_details, video_path, output_path)

        print("*" * 100)

        # match_img_list = [cv2.imread(f'Output_Images_intervals/{match}') for match in top_matches]
        # img_stk = stackImages(match_img_list, 3, 1)
        # cv2.imshow("Matches", img_stk)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main('Query_images', 'Golden_Ratio_features.csv', 'videoplayback (2).mp4', 'Output_Videos')
