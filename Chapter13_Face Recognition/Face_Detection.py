# Import the OpenCV library
import cv2
# Initialize a face cascade using the frontal face haar cascade provided
# with the OpenCV2 library. This will be required for face detection in an
# image.
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# The desired output width and height, can be modified according to the needs.
OUTPUT_SIZE_WIDTH = 700
OUTPUT_SIZE_HEIGHT = 600

# Open the first webcam device
capture = cv2.VideoCapture(0)

# Create two opencv named windows for showing the input, output images.
cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

# Position the windows next to each other
cv2.moveWindow("base-image", 20, 200)
cv2.moveWindow("result-image", 640, 200)

# Start the window thread for the two windows we are using
cv2.startWindowThread()

rectangleColor = (0, 100, 255)

while(1):
    # Retrieve the latest image from the webcam
    rc,fullSizeBaseImage = capture.read()
# Resize the image to 520x420
    baseImage= cv2.resize(fullSizeBaseImage, (520, 420))

# Check if a key was pressed and if it was Q or q, then destroy all
# opencv windows and exit the application, stopping the infinite loop.
    pressedKey = cv2.waitKey(2)
    if (pressedKey == ord('Q')) | (pressedKey == ord('q')):
        cv2.destroyAllWindows()
        exit(0)
# Result image is the image we will show the user, which is a
# combination of the original image captured from the webcam with the
# overlayed rectangle detecting the largest face
    resultImage = baseImage.copy()

# We will be using gray colored image for face detection.
# So we need to convert the baseImage captured by webcam to a gray-based image
    gray_image = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_image, 1.3, 5)
    # As we are only interested in the 'largest' face, we need to
# calculate the largest area of the found rectangle.
# For this, first initialize the required variables to 0.
    maxArea = 0
    x = 0
    y = 0
    w = 0
    h = 0

# Loop over all faces found in the image and check if the area for this face is
# the largest so far
    for(_x, _y, _w, _h) in faces:
        if _w * _h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w * h
# If any face is found, draw a rectangle around the largest face present in the picture
    if maxArea > 0:
        cv2.rectangle(resultImage, (x-10, y-20),(x + w+10, y + h+20), rectangleColor, 2)
# Since we want to show something larger on the screen than the
# original 520x420, we resize the image again

# Note that it would also be possible to keep the large version
# of the baseimage and make the result image a copy of this large
# base image and use the scaling factor to draw the rectangle
# at the right coordinates.
    largeResult = cv2.resize(resultImage,(OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
# Finally, we show the images on the screen
    cv2.imshow("base-image", baseImage)
    cv2.imshow("result-image", largeResult)

