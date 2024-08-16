import cv2
from ultralytics import YOLO

detector = YOLO('yolov8n.pt')

def find_gop(cap):
    c=0
    while (True):
        success, frame = cap.read()
        frame_type = cap.get(cv2.CAP_PROP_FRAME_TYPE)
        if (frame_type == 73):
            c+=1
            if (c==2):
                return cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if (not success):
            break

def write_to_excel(f,data):
    for i in data:
        f.write(str(i)+",")
    f.write("\n")

def extract_all_frames(cap):
    f = open("data_all_frames.csv", "a+")
    f.write("frame_no,object_no,center_x,center_y,width,height\n")
    frame_count = 0
    while (True):
        obj_cnt = 0
        frame_count +=1
        success , frame = cap.read()

        if (not success):
            break

        results = detector(source=frame,save=False)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                (x1,y1,x2,y2) = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # cv2.circle(frame,[int(x),int(y)],5, (0,0,255))
                # cv2.circle(frame,[int(w),int(h)],5,(255,0,0))

                obj_cnt+=1

                write_to_excel(f,[frame_count,obj_cnt,cx,cy,x2-x1,y2-y1])

                cv2.imwrite(f"./Output_Images/frame_{frame_count}_object_{obj_cnt}.jpeg", frame[int(y1):int(y2),int(x1):int(x2)])

        # cv2.imshow("image",frame)
        # cv2.waitKey(1)
    f.close()



def extract_frames_at_interval(cap):
    f = open("data_interval_frames.csv", "a+")
    f.write("frame_no,object_no,time_stamp,center_x,center_y,width,height\n")
    frame_count = 0
    while (True):
        obj_cnt = 0
        frame_count += 1
        success, frame = cap.read()

        if (not success):
            break

        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

        time_stamp = (frame_no) / fps

        if (frame_no %30 == 0):  ## Taking every 30th frame

            results = detector(source=frame, save=False)
            print(frame_count)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    (x1, y1, x2, y2) = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # cv2.circle(frame,[int(x),int(y)],5, (0,0,255))
                    # cv2.circle(frame,[int(w),int(h)],5,(255,0,0))

                    obj_cnt += 1

                    write_to_excel(f, [frame_count, obj_cnt, time_stamp, cx, cy, x2 - x1, y2 - y1])

                    cv2.imwrite(f"./Output_Images_intervals/frame_{frame_count}_object_{obj_cnt}_.jpeg",
                                frame[int(y1):int(y2), int(x1):int(x2)])

            # cv2.imshow("image",frame)
            cv2.waitKey(1)
    f.close()

def extract_i_frames(cap):
    f = open("data_i_frames.csv","a+")
    f.write("frame_no,object_no,time_stamp,center_x,center_y,width,height\n")
    frame_count = 0

    while (True):
        obj_cnt = 0
        frame_count += 1
        success, frame = cap.read()

        if (not success):
            break

        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

        frame_type = cap.get(cv2.CAP_PROP_FRAME_TYPE)

        time_stamp = (frame_no)/fps

        if (frame_no-1)%gop_n == 0 or ((frame_no-gop_n/2))%(gop_n)==0 or (frame_no)%gop_n==0:

            results = detector(source=frame, save=False)
            print(frame_count)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    (x1, y1, x2, y2) = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # cv2.circle(frame,[int(x),int(y)],5, (0,0,255))
                    # cv2.circle(frame,[int(w),int(h)],5,(255,0,0))

                    obj_cnt += 1

                    write_to_excel(f,[frame_count, obj_cnt,time_stamp, cx, cy, x2 - x1, y2 - y1])

                    cv2.imwrite(f"./Output_Images_iframes/frame_{frame_count}_object_{obj_cnt}_.jpeg",
                                frame[int(y1):int(y2), int(x1):int(x2)])

            # cv2.imshow("image",frame)
            cv2.waitKey(1)
    f.close()

PATH = "videoplayback (2).mp4"

cap = cv2.VideoCapture(PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

gop_n = find_gop(cap)

cap.release()

cap = cv2.VideoCapture(PATH)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("GOP: ",gop_n)
print("Total Frames: ",total_frames)
print("FPS: ",fps)

# extract_i_frames(cap)
# extract_all_frames(cap)
extract_frames_at_interval(cap)
