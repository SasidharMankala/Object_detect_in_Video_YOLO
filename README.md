#  Object_detect_in_Video_YOLO

Object detection using YOLO (video stream)

In this project the code will detect objects in realtime using the webcam.
Just run the ```main.py``` file 

This will detect the objects in video stream coming from the webcam

If you want to change the input video stream i.e., instead of webcam using an video located in the local disk we can change the 5th line from ```cap = cv2.VideoCapture(0)``` to ```cap = cv2.imread('filepath')
