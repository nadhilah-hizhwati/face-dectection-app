import streamlit as st
import pandas as pd
import requests
import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from datetime import datetime
import zipfile
import re

# Title
st.title("🕵️‍♂️ Face Detection App")
st.write("👩‍🚀 Please contact nadhilah.hizhwati@amartha.com for more detail 📰 / report bug 🐞")

# Initialize session state
if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "detected_images" not in st.session_state:
    st.session_state.detected_images = None
if "undetected_images" not in st.session_state:
    st.session_state.undetected_images = None

# 1. User Input
uploaded_excel = st.file_uploader("📄 Upload file Excel yang berisi URL gambar (kolom harus bernama 'URL')", type=["xlsx"])
index_row_awal = st.number_input("🔢 Index baris awal (mulai dari 0)", min_value=0, value=0)
index_row_akhir = st.number_input("🔢 Index baris akhir (inklusif)", min_value=0, value=0)

start_process = st.button("🚀 Jalankan Face Detection")

# 2. Face Detection Logic
if start_process:
    if uploaded_excel:
        df_urls = pd.read_excel(uploaded_excel)

        if 'URL' not in df_urls.columns:
            st.error("❌ File Excel harus memiliki kolom bernama 'URL'")
        elif index_row_awal > index_row_akhir or index_row_akhir >= len(df_urls):
            st.error("❌ Index awal/akhir tidak valid.")
        else:
            df_selected = df_urls.iloc[index_row_awal:index_row_akhir + 1].copy()
            st.success(f"✅ Memproses {len(df_selected)} gambar dari index {index_row_awal} hingga {index_row_akhir}.")

            df_result = pd.DataFrame(columns=['Original URL', 'Face Array', 'Filename'])

            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            progress_bar = st.progress(0)
            status_text = st.empty()

            detected_images = []
            undetected_images = []

            for idx, (i, row) in enumerate(df_selected.iterrows()):
                image_url = row['URL']
                try:
                    status_text.text(f"🔄 Processing: {image_url}")

                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content)).convert('RGB')

                    # Check if image is horizontal
                    if img.width > img.height:
                        img = img.rotate(90, expand=True)

                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    def detect_faces(image_cv):
                        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)
                        faces = face_classifier.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
                        )
                        return faces

                    faces = detect_faces(img_cv)

                    # Jika rotate pertama tidak berhasil, coba rotate tambahan 90 derajat lagi
                    if len(faces) == 0:
                        img = img.rotate(90, expand=True)
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        faces = detect_faces(img_cv)

                    # Draw rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 4)

                    # Sanitize filename from URL
                    safe_filename = re.sub(r'[^A-Za-z0-9]+', '_', image_url)
                    filename = f"result_{safe_filename}.jpg"

                    is_detected = len(faces) > 0

                    # Convert processed image to memory buffer
                    is_success, buffer = cv2.imencode(".jpg", img_cv)
                    img_bytes = BytesIO(buffer.tobytes())

                    if is_detected:
                        detected_images.append((filename, img_bytes.getvalue()))
                    else:
                        undetected_images.append((filename, img_bytes.getvalue()))

                    df_result.loc[i] = [image_url, str(faces), filename]

                except Exception as e:
                    st.warning(f"⚠️ Gagal memproses index {i}: {e}")

                progress_bar.progress((idx + 1) / len(df_selected))

            st.success("✅ Face detection selesai.")

            # Simpan ke session_state supaya tidak hilang saat app rerun
            st.session_state.df_result = df_result
            st.session_state.detected_images = detected_images
            st.session_state.undetected_images = undetected_images

    else:
        st.error("Mohon upload file Excel dan isi semua input dengan benar.")

# 3. Setelah Face Detection selesai (bagian download)
if st.session_state.df_result is not None:

    st.subheader("📂 Download Result Files")

    st.dataframe(st.session_state.df_result)

    # Download CSV
    csv = st.session_state.df_result.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download hasil (CSV)", data=csv, file_name="face_detection_result.csv", mime="text/csv")

    # Function to create ZIP
    def create_zip_file(image_list):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, content in image_list:
                zip_file.writestr(name, content)
        zip_buffer.seek(0)
        return zip_buffer

    # Download Detected ZIP
    if st.session_state.detected_images:
        zip_detected = create_zip_file(st.session_state.detected_images)
        st.download_button(
            "🌞 Download All Detected Faces (ZIP)",
            data=zip_detected,
            file_name="detected_faces.zip",
            mime="application/zip"
        )

    # Download Undetected ZIP
    if st.session_state.undetected_images:
        zip_undetected = create_zip_file(st.session_state.undetected_images)
        st.download_button(
            "😶‍🌫️ Download All Undetected Faces (ZIP)",
            data=zip_undetected,
            file_name="undetected_faces.zip",
            mime="application/zip"
        )
