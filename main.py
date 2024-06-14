import cv2
import numpy as np
import autopy
import mediapipe as mp

class Tracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        
        self.calibration = None


    def getGazePosition(self, image):
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_indices = list(range(362, 382))
                right_eye_indices = list(range(133, 153))
                left_eye = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye = [face_landmarks.landmark[i] for i in right_eye_indices]
                
                left_eye_x = np.mean([mark.x for mark in left_eye]) * image.shape[1]
                left_eye_y = np.mean([mark.y for mark in left_eye]) * image.shape[0]
                right_eye_x = np.mean([mark.x for mark in right_eye]) * image.shape[1]
                right_eye_y = np.mean([mark.y for mark in right_eye]) * image.shape[0]
                
                x = (left_eye_x + right_eye_x) / 2
                y = (left_eye_y + right_eye_y) / 2
                return (x, y)
        return
    

    def calibrate(self, image):
        gaze_pos = self.getGazePosition(image)
        if gaze_pos:
            self.calibration = gaze_pos


def main():
    smooth = 10.0 # Smoothing factor
    sens = 20.0 # Sensitivity
    deadzone = 0.005

    screen_w, screen_h = autopy.screen.size()
    cap = cv2.VideoCapture(0)
    tracker = Tracker()
    prev_x, prev_y = screen_w // 2, screen_h // 2

    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.calibrate(frame_rgb)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gaze_pos = tracker.getGazePosition(frame_rgb)

        if gaze_pos is not None and tracker.calibration is not None:
            x, y = gaze_pos
            cal_x, cal_y = tracker.calibration
            
            delta_x = (cal_x - x) / frame.shape[1]
            delta_y = (y - cal_y) / frame.shape[0]
            
            if abs(delta_x) < deadzone:
                delta_x = 0
            
            if abs(delta_y) < deadzone:
                delta_y = 0
            
            delta_x = np.sign(delta_x) * (abs(delta_x) ** 1.1) * sens
            delta_y = np.sign(delta_y) * (abs(delta_y) ** 1.1) * sens
            
            mapped_x = prev_x + delta_x * (screen_w / 4)
            mapped_y = prev_y + delta_y * (screen_h / 4)
            
            mapped_x = max(0, min(mapped_x, screen_w))
            mapped_y = max(0, min(mapped_y, screen_h))
            
            curr_x = prev_x + (mapped_x - prev_x) / smooth
            curr_y = prev_y + (mapped_y - prev_y) / smooth

            autopy.mouse.move(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        cv2.imshow("Archead", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()