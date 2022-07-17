import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

vc = cv2.VideoCapture(0)
def draw_landmark(frame, results):
    height, width, _ = frame.shape
    if results.multi_face_landmarks is None:
        return frame
    for facial_landmarks in results.multi_face_landmarks:
        for i in range(50, 200):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            cv2.circle(frame, (x, y), 2, (100, 100, 0), -1)
    return frame
while cv2.waitKey(1) < 0:
    grabbed, frame = vc.read()
    print(grabbed)
    if not grabbed:
        exit() 
    face_mesh = mp_face_mesh.FaceMesh()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    print(results.multi_face_landmarks)
    height, width, _ = frame.shape
    frame = draw_landmark(frame, results)
    cv2.imshow('FaceMesh', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()