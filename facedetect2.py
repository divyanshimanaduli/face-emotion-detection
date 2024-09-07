import cv2
import numpy as np
import keras

facedetect = cv2.CascadeClassifier(r'C:\Users\Divyanshi\Desktop\facedetect\haarcascade_frontalface_default.xml')

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(7, activation='softmax')
])

model.load_weights(r'model.h5')

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

video = cv2.VideoCapture(0)

while video.isOpened():
    check, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
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
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
