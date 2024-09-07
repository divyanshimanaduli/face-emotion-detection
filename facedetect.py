import cv2
import numpy as np
import tensorflow as tf

facedetect = cv2.CascadeClassifier(r'C:\Users\Divyanshi\Desktop\facedetect\haarcascade_frontalface_default.xml')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.load_weights(r'model.h5')

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

frame = cv2.imread(r"C:\Users\Divyanshi\Desktop\facedetect\Screenshot 2024-01-20 072415.png")
#frame=cv2.imread(r"C:\Users\Divyanshi\Desktop\facedetect\happy.jpg")
#frame=cv2.imread(r"C:\Users\Divyanshi\Desktop\facedetect\angry.jpg")
#frame=cv2.imread(r"C:\Users\Divyanshi\Desktop\facedetect\Screenshot 2024-01-20 072158.png")


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = facedetect.detectMultiScale(gray, 1.3, 3)

for x, y, w, h in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
    face = gray[y:y+h, x:x+w]
    rsz = cv2.resize(face, (48, 48))
    normalize = rsz / 255.0
    reshape = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshape)
    label = np.argmax(result, axis=1)[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, labels[label], (x, y), font, 0.8, (0, 255, 255), 4)

frame = cv2.resize(frame, (400, 400))
cv2.imshow('output', frame)
cv2.waitKey(0)
