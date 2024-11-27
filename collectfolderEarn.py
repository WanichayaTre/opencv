import cv2
import numpy as np
import os

# โหลดโมเดลตรวจจับใบหน้า
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# กำหนดโฟลเดอร์ที่มีรูปภาพต้นฉบับ และโฟลเดอร์สำหรับบันทึกผลลัพธ์
input_folder = "earnface"
output_folder = "processed_earn"
os.makedirs(output_folder, exist_ok=True)

# ฟังก์ชันบันทึกภาพ
def save_image(face_img, user_id, img_count, effect_name):
    filename = f"User.{user_id}.{img_count}_{effect_name}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, face_img)
    print(f"Saved: {filename}")

# ฟังก์ชันประมวลผลภาพ
def apply_effects(face_img):
    effects = {}

    # Original
    effects['original'] = face_img

    # Noise
    noise = np.random.normal(0, 25, face_img.shape).astype(np.uint8)
    effects['noise'] = cv2.add(face_img, noise)

    # Blur
    effects['blur'] = cv2.GaussianBlur(face_img, (5, 5), 0)

    # Threshold
    _, effects['threshold'] = cv2.threshold(face_img, 127, 255, cv2.THRESH_BINARY)

    # Brightness
    effects['brightness'] = cv2.convertScaleAbs(face_img, beta=50)

    # Contrast
    effects['contrast'] = cv2.convertScaleAbs(face_img, alpha=1.5)

    return effects

# อ่านรูปจากโฟลเดอร์และประมวลผล
current_id = 1
image_count = 1
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(input_folder, filename)
        frame = cv2.imread(filepath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            # ใช้ฟังก์ชันปรับแต่งเอฟเฟกต์
            effects = apply_effects(face_img)

            # บันทึกภาพแต่ละเอฟเฟกต์
            for effect_name, effect_img in effects.items():
                save_image(effect_img, current_id, image_count, effect_name)

            image_count += 1

print("Image Processing Done")
