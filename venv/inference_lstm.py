
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import keras.models
import threading

# đọc ảnh
cap = cv2.VideoCapture(0)

# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# load model
model = keras.models.load_model("model.h5")

lm_list = []
#label = "BODYSWING"
#no_of_frames = 600

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # vẽ các điểm
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
    return img    

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0,255,0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis = 0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    #print(results)
    if results[0][0] > 0.5:
        label = "Swing Body"
    else:
        label = "Swing Hand"
    return label    


label = "..."

i=0
warmup_frames = 60

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i+1
    if i > warmup_frames:
        print("Start detect...")
        
        if results.pose_landmarks:
            # ghi nhận thông số khung sương
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)

            if len(lm_list) == 10:

                # đưa vào nhận diên
                thread_01 = threading.Thread(target = detect, args = (model, lm_list,))
                thread_01.start()
                lm_list = []
                # ghi kết quả lên ảnh

            # vẽ khung xương lên ảnh
            img = draw_landmark_on_image(mpDraw, results, img)

        img = draw_class_on_image(label, img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

#ghi file csv

#df = pd.DataFrame(lm_list)
#df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindowns()