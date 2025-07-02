# 1. Importación de librería
import cv2 # esta libreria nos permite leer videos y imagenes a tiempo real

# 2. Clasificador Haar para rostros y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# el face_cascade es para detectar el rostro 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# el eye_cascade es para la deteccion de ojos
#### y estas dos linias traen el modelo Har que opencv trae por defecto 

# 3. Activar la cámara web
cap = cv2.VideoCapture(0)

# 4. Bucle principal de captura de video
while True:
    ret, frame = cap.read() 

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6. Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces: # este codigo recorre cada rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # y en esta parte dibuja un rectangulo de color verde a nuestro rostro

        # 7. Región de  para buscar ojos solo dentro del rostro
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # 8. Detectar ojos dentro de la región del rostro
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        # scaleFactor es para detectar con presicion 

        for (ex, ey, ew, eh) in eyes:
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            radius = int((ew + eh) * 0.25)
            cv2.circle(roi_color, (center_x, center_y), radius, (255, 0, 0), 2)
            # El color (255,0,0) es azul en formato BGR de OpenCV.

    # 9. Mostrar el video en vivo
    cv2.imshow('Detector de Ojos', frame)

    # 10. tecla para salir 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11. Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
