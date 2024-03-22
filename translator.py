import cv2
import mediapipe as mp
import pickle

import numpy as np

model_dic = pickle.load(open('./model.p', 'rb'))
model = model_dic['model']

cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
              13: 'n', 14: '0', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
              24: 'X', 25: 'Y', 26: 'Z'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = [lm for lm in hand_landmarks.landmark]
            data_aux = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]  # Flatten x and y

            # Ensure landmarks is a 2D array with a single sample, with padding if necessary
            if len(data_aux) < 500:
                # Pad data_aux to match the expected 500-feature input size
                data_aux_padded = np.pad(data_aux, (0, 500 - len(data_aux)), 'constant')
                data_aux_reshaped = np.array(data_aux_padded).reshape(1, -1)  # Reshape to 2D array for sklearn model
            else:
                data_aux_reshaped = np.array(data_aux).reshape(1, -1)

            # Predict using the model
            prediction = model.predict(data_aux_reshaped)
            predicted_label = labels_dic[int(prediction[0])]

            # Display the prediction on the frame
            cv2.putText(frame, predicted_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
