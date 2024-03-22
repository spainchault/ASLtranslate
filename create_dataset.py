import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  # Create the full path of the directory
    if os.path.isdir(dir_path):  # Check if it's a directory
        for img_path in os.listdir(dir_path):
            full_img_path = os.path.join(dir_path, img_path)
            if not os.path.isfile(full_img_path):  # Ensure it's a file before processing
                continue
            img = cv2.imread(full_img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)  # This might add the directory name as label

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
