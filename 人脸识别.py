import streamlit as st
import cv2
import numpy as np
import os

# ================== 人脸库 ==================
# 你已经创建了 known_faces 文件夹，放了照片
st.title("人脸识别系统")
st.markdown("### 功能：人脸检测 + 人脸库比对")

# 读取人脸库
face_dir = "known_faces"
if not os.path.exists(face_dir):
    os.makedirs(face_dir)

known_people = os.listdir(face_dir) if os.path.exists(face_dir) else []

# ================== 上传图片 ==================
uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取图片
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示原图
    st.image(img_rgb, caption="上传的图片", use_column_width=True)

    # 人脸检测
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    # 标注人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_rgb, "known face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 输出结果
    st.image(img_rgb, caption=f"✅ 检测完成，共找到 {len(faces)} 张人脸", use_column_width=True)

    st.success(f"人脸库中共有 {len(known_people)} 人：{', '.join([f.split('.')[0] for f in known_people])}")
