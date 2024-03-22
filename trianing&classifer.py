import os
import pickle
import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Assuming 'data' and 'labels' are correctly loaded and populated
# This is a placeholder; you need to ensure data and labels are correctly filled according to your dataset structure
data = []
labels = []

DATA_DIR = './data'
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv.imread(img_path)
            if img is not None:
                resized_img = cv.resize(img, (128, 128))
                # Flatten the image and add it to 'data'
                data.append(resized_img.flatten())
                # Use directory names as labels
                labels.append(dir_)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Now you can split your data
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()