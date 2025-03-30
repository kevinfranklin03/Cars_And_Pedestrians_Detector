import cv2
# print(cv2.__version__)
img_file = '../images/car.jpg'

# opencv image read
img = cv2.imread(img_file)
# video = cv2.VideoCapture('../videos/dashcam.mp4')
video = cv2.VideoCapture('../videos/pedes.mp4')

# Convert to grayscale(for haar cascade!)
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Classifier files!
car_file = '../utils/car_detector.xml'
pedestrian_file = '../utils/pedestrians.xml'

# Car Classifier
car_tracker = cv2.CascadeClassifier(car_file)
# pedestrian Classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)

# run until a car stops
while True:
    # getting frame from a video
    (read_successful, frame) = video.read()
    frame = cv2.resize(frame, (640, 360))

    if read_successful:
        # convert to greyscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)
    # print(cars)

    # Draw boxes around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw boxes around the cars
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('car and pedestrians detection', frame)
    # cv2.imshow('black_n_white', black_n_white)
    #
    cv2.waitKey(1)







# cv2.imshow('car detection',img)
# # cv2.imshow('black_n_white', black_n_white)
#
# cv2.waitKey()

print("Code Completed")