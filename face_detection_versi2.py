import streamlit as st
import os
import pandas as pd
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

# Title
st.title("🕵️‍♂️ Face Detection App")
st.write("👩‍🚀 Please contact nadhilah.hizhwati@amartha.com for more detail 📰 / report bug 🐞")
# 1. User Input
uploaded_excel = st.file_uploader("📄 Upload file Excel yang berisi URL gambar (kolom harus bernama 'URL')", type=["xlsx"])
save_detected_result_path = st.text_input("📥 Folder untuk menyimpan foto dengan wajah terdeteksi:")
save_undetected_result_path = st.text_input("📥 Folder untuk menyimpan foto tanpa wajah terdeteksi:")

index_row_awal = st.number_input("🔢 Index baris awal (mulai dari 0)", min_value=0, value=0)
index_row_akhir = st.number_input("🔢 Index baris akhir (inklusif)", min_value=0, value=0)

start_process = st.button("🚀 Jalankan Face Detection")

# 2. Face Detection Logic
if start_process:
    if uploaded_excel and save_detected_result_path and save_undetected_result_path:
        os.makedirs(save_detected_result_path, exist_ok=True)
        os.makedirs(save_undetected_result_path, exist_ok=True)

        df_urls = pd.read_excel(uploaded_excel)

        if 'URL' not in df_urls.columns:
            st.error("❌ File Excel harus memiliki kolom bernama 'URL'")
        else:
            if index_row_awal > index_row_akhir:
                st.error("❌ Index awal tidak boleh lebih besar dari index akhir.")
            elif index_row_akhir >= len(df_urls):
                st.error(f"❌ Index akhir melebihi jumlah data. Jumlah baris: {len(df_urls)}")
            else:
                df_selected = df_urls.iloc[index_row_awal:index_row_akhir + 1].copy()
                st.success(f"✅ Memproses {len(df_selected)} gambar dari index {index_row_awal} hingga {index_row_akhir}.")

                df_result = pd.DataFrame(columns=[
                    'Original URL', 'Face Array', 'Saved Path'
                ])

                face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                progress_bar = st.progress(0)

                for i, row in df_selected.iterrows():
                    image_url = row['URL']
                    try:
                        response = requests.get(image_url, timeout=10)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)
                        faces = face_classifier.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
                        )

                        filename = f"result_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        if len(faces) > 0:
                            save_path = os.path.join(save_detected_result_path, filename)
                        else:
                            save_path = os.path.join(save_undetected_result_path, filename)

                        for (x, y, w, h) in faces:
                            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 4)

                        cv2.imwrite(save_path, img_cv)
                        df_result.loc[i] = [image_url, str(faces), save_path]

                    except Exception as e:
                        st.warning(f"Gagal memproses index {i}: {e}")

                    progress_bar.progress((i - index_row_awal + 1) / len(df_selected))

                st.success("✅ Face detection selesai.")
                st.dataframe(df_result)

                # Optional: Tambahkan tombol untuk download CSV hasil
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download hasil (CSV)", data=csv, file_name="face_detection_result.csv", mime="text/csv")
    else:
        st.error("Mohon upload file Excel dan isi semua input dengan benar.")
