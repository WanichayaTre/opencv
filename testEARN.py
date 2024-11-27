import cv2
import os

# โหลด Cascade Classifier สำหรับการตรวจจับใบหน้า
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# โหลดตัวจดจำใบหน้า LBPH จากไฟล์ที่บันทึกไว้
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainEarn.yml")

# ลิสต์ของชื่อที่สอดคล้องกับ serial ของใบหน้า
name_list = ["", "Earn"]  # "" คือตัวแทนสำหรับ serial 0 (ไม่ได้กำหนดชื่อ)

# กำหนด path ของภาพที่ต้องการประมวลผล
input_path = "EARN.jpg"

# อ่านและประมวลผลภาพ
if os.path.exists(input_path):
    frame = cv2.imread(input_path)

    # เปลี่ยนขนาดภาพให้เป็น 640x480
    frame = cv2.resize(frame, (640, 480))

    # แปลงเป็นภาพสีเทา
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # ทำนายใบหน้าที่ตรวจจับได้
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # กำหนดชื่อและความเชื่อมั่น
        label = "Unknown"
        if conf > 50:  # ปรับเกณฑ์ความเชื่อมั่น
            label = name_list[serial]

        # วาดกรอบและใส่ชื่อบนภาพ
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # แสดงภาพที่ประมวลผล
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(0)  # กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง
    cv2.destroyAllWindows()

else:
    print(f"Error: File not found at {input_path}")
