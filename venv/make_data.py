
import cv2
import mediapipe as mp
import pandas as pd

# đọc ảnh
cap = cv2.VideoCapture(0)

# khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

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
while True:
    ret, frame = cap.read()
    if ret:
        # nhận diện pose
        framRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(framRGB)

        if results.pose_landmarks:
            # ghi nhận thông số khung sương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

#ghi file csv

#df = pd.DataFrame(lm_list)
#df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindowns()