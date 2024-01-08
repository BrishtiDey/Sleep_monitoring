from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import vlc

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
# Set resolutions of frame.
# convert from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create VideoWriter object.
# and store the output in 'captured_video.avi' file.

video_cod = cv2.VideoWriter_fourcc(*'XVID')
video_output= cv2.VideoWriter('../../../Automated_car_features/Drowsiness_alarm/cca.avi',
                      video_cod,
                      10,
                      (frame_width,frame_height))

while True:
	ret, frame=cap.read()
	# if ret == True: 
	
	#frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:

				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,465),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				if flag % 10 == 0: 
					p = vlc.MediaPlayer('1.wav')
					p.play()
		else:
			flag = 0
	# Write the frame into the file 'captured_video.avi'
	video_output.write(frame)

	# Display the frame, saved in the file   
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
