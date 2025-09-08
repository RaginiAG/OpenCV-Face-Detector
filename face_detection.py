import cv2

Face_cap = cv2.CascadeClassifier("C:/Users/LENOVO/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Open the default webcam
video_cap = cv2.VideoCapture(0)

# Create a window that can be resized
cv2.namedWindow("video_live", cv2.WINDOW_NORMAL)


while True:
    ret, video_data = video_cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for mirror effect
    video_data = cv2.flip(video_data, 1)

    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    Faces = Face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in Faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the video feed in the window
    cv2.imshow("video_live", video_data)
    
    if cv2.waitKey(10) & 0xFF == ord("a"):  # press 'a' to exit
        break
# Release the webcam resource
video_cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
