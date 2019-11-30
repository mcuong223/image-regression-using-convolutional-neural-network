import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    model = None
    try:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model('./trained-model/cnn1.h5', compile=False)
    except Exception as ex:
        print(ex)
        pass
    if model == None:
        print('Could not load the model')
        return
    
    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # Define the upper and lower boundaries for a color to be considered "Blue"
    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])

    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Define filters
    filters = [
        './images/sunglasses_5.jpg', 
        './images/sunglasses_2.png', 
        './images/sunglasses_4.png', 
        './images/sunglasses.png', 
        './images/sunglasses_3.jpg', 
        './images/sunglasses_6.png'
        ]
    filterIndex = 0

    # Load the video - O for webcam input
    camera = cv2.VideoCapture(0)

    camera.set(3,500)
    camera.set(4,500)
    # camera.sleep(2)
    # cap.set(15, -8.0)
    # Keep reading the input
    while True:
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        frame2 = np.copy(frame)
        frame = frame.astype('uint8')
        # Convert to HSV and GRAY for convenience
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces using the haar cascade object
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)
        # Add the 'Next Filter' button to the frame
        # frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)
        # cv2.putText(frame, "NEXT FILTER", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Determine which pixels fall within the blue boundaries 
        # and then blur the binary image to remove noise
        # blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        # blueMask = cv2.erode(blueMask, kernel, iterations=2)
        # blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        # blueMask = cv2.dilate(blueMask, kernel, iterations=1)

        # # Find contours (bottle cap in my case) in the image
        # (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # center = None

        # # Check to see if any contours were found
        # if len(cnts) > 0:
        #     # Sort the contours and find the largest one -- we
        #     # will assume this contour correspondes to the area of the bottle cap
        #     cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        #     # Get the radius of the enclosing circle around the found contour
        #     ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        #     # Draw the circle around the contour
        #     cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        #     # Get the moments to calculate the center of the contour (in this case Circle)
        #     M = cv2.moments(cnt)
        #     center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #     if center[1] <= 65:
        #         if 500 <= center[0] <= 620: # Next Filter
        #             filterIndex += 1
        #             filterIndex %= 6
        #             continue
        # Loop over all the faces found in the frame
        for (x, y, w, h) in faces:
            # Make the faces ready for the model (normalize, resize and stuff)
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+w]

            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255

            # Resize it to 96x96 to match the input format of the model
            original_shape = gray_face.shape # A Copy for future reference
            face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_copy = face_resized.copy()
            face_resized = face_resized.reshape(1, 96, 96, 1)

            # Predict the keypoints using the model
            keypoints = model.predict(face_resized)

            # De-Normalize the keypoints values
            keypoints = keypoints * 48 + 48

            # Map the Keypoints back to the original image
            face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_color2 = np.copy(face_resized_color)

            # Pair the keypoints together - (x1, y1)
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))

             # Add FILTER to the frame
            sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

            # Map the face with shades back to its original shape
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

            # Add KEYPOINTS to the frame2
            for keypoint in points:
                cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)
            
            # Map the face with keypoints back to the original image (a separate one)
            frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)
            # Show the frame and the frame2
            cv2.imshow("Selfie Filters", frame)
            # cv2.imshow("Facial Keypoints", frame2)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break




if __name__ == "__main__":
    main()