import streamlit as st
import cv2
import numpy as np

# 网页标题
st.title("人脸识别工具（精准版）")
st.subheader("只框真正的人脸，不乱框！")

# 上传图片
uploaded_file = st.file_uploader("请上传清晰正脸照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示原图
    st.image(img_rgb, caption="原图", use_column_width=True)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ 这是【精准模式】，只框真脸，不乱框！
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,  # 更严格
        minNeighbors=8,  # 更严格
        minSize=(80, 80)  # 只框大脸，忽略小东西
    )

    # 画绿色框
    for (x, y, w, h) in faces:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 显示结果
    st.image(img_rgb, caption=f"✅ 精准检测到 {len(faces)} 张人脸", use_column_width=True)