import numpy as np
import face_recognition as fr
import cv2
import os
import pygame

pygame.init()
video_capture = cv2.VideoCapture(0)
known_face_names=[]
known_face_encodings=[]
known_dir="C:\\Users\\Satyam\\Documents\\flask_app\\pictures"
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
X = 400
Y = 400
display_surface = pygame.display.set_mode((X, Y))
font = pygame.font.Font('freesansbold.ttf', 32)
for file in os.listdir(known_dir):
    img = cv2.imread(known_dir+"\\"+file)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    img2=cv2.resize(img, (width, height))
    b_face_encoding = fr.face_encodings(img2)[0]
    known_face_names.append(file.split('.')[0])
    known_face_encodings.append(b_face_encoding)
    

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
        

    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        text = font.render(name, True, green, blue)
        textRect = text.get_rect()
        textRect.center = (X // 2, Y // 2)


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        display_surface.fill(white)
        display_surface.blit(text, textRect)
        pygame.display.update()

    cv2.imshow('Webcam_facerecognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()