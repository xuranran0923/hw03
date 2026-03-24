import streamlit as st
import cv2
import numpy as np
import os

# 网页标题
st.title("人脸识别工具（精准版）")
st.subheader("只框真正的人脸，不乱框！")
st.markdown("### 功能：人脸检测 + 人脸库比对")

# ================== 人脸库 ==================
face_dir = "known_faces"
if not os.path.exists(face_dir):
    os.makedirs(face_dir)

known_people = os.listdir(face_dir) if os.path.exists(face_dir) else []

# ================== 上传图片 ==================
uploaded_file = st.file_uploader("请上传清晰正脸照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ 只读取一次文件（关键修复）
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 转RGB用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示原图
    st.image(img_rgb, caption="上传的原图", use_column_width=True)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ 精准检测参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(80, 80)
    )

    # 画框 + 标注
    for (x, y, w, h) in faces:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_rgb, "face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示结果
    st.image(img_rgb, caption=f"✅ 精准检测到 {len(faces)} 张人脸", use_column_width=True)
    st.success(f"人脸库中共有 {len(known_people)} 人：{', '.join([f.split('.')[0] for f in known_people])}")
    # 输出结果
    st.image(img_rgb, caption=f"✅ 检测完成，共找到 {len(faces)} 张人脸", use_column_width=True)

    st.success(f"人脸库中共有 {len(known_people)} 人：{', '.join([f.split('.')[0] for f in known_people])}")
