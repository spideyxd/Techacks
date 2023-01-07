from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import math as m

app = Flask(__name__)

# use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:


        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # capturing webcam
        cap = cv2.VideoCapture(0)


        # detection
        # for lndmrk in mp_pose.PoseLandmark:
        #     print(lndmrk)


        # Calculating angle
        counter = 0
        incorrect_rep = 0
        stage = None
        stage2 = None


        def angle_calc(a, b, c):  # a is 11   b is 13 and   c is 15

            # converting them into numpy
            a = np.array(a)  # shoulder to elbow
            b = np.array(b)  # elbow
            c = np.array(c)  # elbow to wrist

            radians = np.arctan2(c[1]-b[1], c[0] - b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
            # distance formula
            #  y-coordinates             c[1] = y2 b[1] = y1 ,
            #  x-coordinates             c[0] = x2  b[0] = x1
            angle = np.abs(radians*180.0/np.pi)
            if angle > 180.0:
                angle = 360-angle
            return angle


        def findDistance(a, b, h):
            a = np.array(a)  # shoulder to elbow
            b = np.array(b)
            h = np.array(h)
            s_to_e = np.arctan2(a[1]-b[1], a[0]-b[0])
            s_to_h = np.arctan2(a[1]-h[1], a[0] - h[0])
            dist = s_to_e - s_to_h
            return dist


        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

            while cap.isOpened():
                ret, frame = cap.read()

            # rendor stuff
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # make detection
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # extract landmarks

                # using try and except if something wrong happens then the program will not get terminated instead it would get pass.

                try:
                    landmarks = results.pose_landmarks.landmark

                    # coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # angle Calculation
                    ang = angle_calc(shoulder, elbow, wrist)
                    angle = round(ang, 2)

                    # showing on screen
                    cv2.rectangle(image,  (0, 0), (220, 100), (255, 128, 0), -1)
                    cv2.putText(image, "Angle: " + str(stage), tuple(np.multiply(elbow, [1140, 780]).astype(int)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    dist = findDistance(shoulder, elbow, left_hip)
                    dist_round = round(dist, 2)
                    # Repetition counter

                    if angle > 165:
                        stage = "down"
                    if angle < 40 and stage == "down":
                        stage = "up"
                        counter += 1

                    if dist_round > 0.2:

                        cv2.rectangle(image, (0, 0), (220, 100), (220, 20, 60), -1)

                        cv2.putText(image, "Status: Incorrect Posture", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2,
                                    cv2.LINE_AA)

                    else:
                        cv2.rectangle(image, (0, 0), (220, 100), (255, 128, 0), -1)
                        cv2.putText(image, "Status: Correct Posture", (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2,
                                    cv2.LINE_AA)

                except:
                    pass

                # showing detection
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # showing on screen

                cv2.putText(image, "Repititions: "+str(counter), (5, 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Webcam", image)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        print(angle_calc(shoulder, elbow, wrist))


        l_shldr_x = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        l_shldr_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)

        l_hip_x = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x)
        l_hip_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)


        print(findDistance(shoulder, elbow, left_hip))


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/", methods=['GET', 'POST'])
def index():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Start') == 'Start':
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            # pass # unknown
            return render_template("index.html")
    elif request.method == 'GET':
        # return render_template("index.html")
        print("No Post Back Call")
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
