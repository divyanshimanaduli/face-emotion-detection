hile video.isOpened():
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
