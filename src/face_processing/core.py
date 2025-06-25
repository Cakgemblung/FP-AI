from deepface import DeepFace
import tempfile
import os
import numpy as np
from PIL import Image
import io
import cv2

# Konstanta yang tidak diubah oleh pengguna
# VERIFY_DISTANCE_METRIC DIHAPUS DARI SINI, AKAN JADI PARAMETER
ANALYZE_ACTIONS = ['age', 'emotion', 'gender', 'race']

def verify_images(img1_bytes, img2_bytes, model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine"): # TAMBAHKAN distance_metric
    """
    Memverifikasi similaritas antara dua gambar wajah menggunakan model, detektor, dan metrik jarak yang dipilih.
    """
    img1_path, img2_path = None, None
    results = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            tmp1.write(img1_bytes)
            img1_path = tmp1.name
            tmp2.write(img2_bytes)
            img2_path = tmp2.name

        results = DeepFace.verify(
            img1_path=img1_path, img2_path=img2_path, model_name=model_name,
            detector_backend=detector_backend, distance_metric=distance_metric, # GUNAKAN PARAMETER
            enforce_detection=False, align=False, silent=True
        )
        results['model_name_used'] = model_name
        results['detector_backend_used'] = detector_backend
        results['distance_metric_used'] = distance_metric # Tambahkan metrik yang digunakan ke hasil
        return results
    except Exception as e:
        error_msg = f"Verifikasi gagal (Model: {model_name}, Det: {detector_backend}, Metrik: {distance_metric}): {type(e)._name_} - {str(e)}"
        return {"error": error_msg, "model_name_used": model_name, "detector_backend_used": detector_backend, "distance_metric_used": distance_metric}
    finally:
        if img1_path and os.path.exists(img1_path): os.remove(img1_path)
        if img2_path and os.path.exists(img2_path): os.remove(img2_path)

# --- Fungsi analyze_face_attributes dan extract_aligned_face_bytes tetap SAMA seperti versi terakhir ---
# --- Saya tidak akan menyalinnya lagi untuk menghemat ruang, pastikan Anda menggunakan versi terakhirnya ---
def analyze_face_attributes(img_bytes, detector_backend="opencv"):
    """Menganalisis atribut wajah dari gambar (idealnya sudah di-crop dan align)."""
    img_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_bytes)
            img_path = tmp.name

        detected_faces_data = DeepFace.analyze(
            img_path=img_path, actions=ANALYZE_ACTIONS, detector_backend=detector_backend,
            enforce_detection=False, align=False, silent=True
        )
        if not detected_faces_data:
            return {"data": [], "error": "Analisis atribut tidak menghasilkan data (wajah tidak valid?)."}
        for face_info in detected_faces_data:
            face_info['detector_backend_used_for_attributes'] = detector_backend
        return {"data": detected_faces_data, "error": None}
    except Exception as e:
        error_msg = f"Analisis atribut gagal (Detektor: {detector_backend}): {type(e)._name_} - {str(e)}"
        return {"data": [], "error": error_msg}
    finally:
        if img_path and os.path.exists(img_path): os.remove(img_path)

def extract_aligned_face_bytes(img_bytes_original, detector_backend="opencv"):
    """Mendeteksi, meng-crop, dan meng-align wajah dari gambar asli."""
    img_path_original = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
            tmp_orig.write(img_bytes_original)
            img_path_original = tmp_orig.name

        extracted_face_data_list = DeepFace.extract_faces(
            img_path=img_path_original, detector_backend=detector_backend,
            enforce_detection=True, align=True
        )

        if extracted_face_data_list and len(extracted_face_data_list) > 0:
            face_numpy_float = extracted_face_data_list[0]['face']
            face_numpy_uint8 = (face_numpy_float * 255).astype(np.uint8)
            # Jika versi core.py sebelumnya yang memperbaiki warna sudah benar, gunakan itu.
            # Jika extract_faces mengembalikan BGR, maka konversi ke RGB diperlukan:
            # face_numpy_rgb_uint8 = cv2.cvtColor(face_numpy_uint8, cv2.COLOR_BGR2RGB)
            # pil_image_cropped = Image.fromarray(face_numpy_rgb_uint8)
            # Jika extract_faces sudah RGB (seperti yang kita simpulkan terakhir):
            pil_image_cropped = Image.fromarray(face_numpy_uint8)


            img_byte_arr = io.BytesIO()
            pil_image_cropped.save(img_byte_arr, format='PNG') # PNG untuk debug, bisa JPEG
            img_byte_arr = img_byte_arr.getvalue()
            
            original_region = extracted_face_data_list[0].get('facial_area')
            return {"face_bytes": img_byte_arr, "original_region": original_region, "error": None}
        else:
            return {"face_bytes": None, "original_region": None, "error": "Tidak ada wajah yang dapat di-extract."}
    except Exception as e:
        error_msg = f"Proses extract wajah gagal (Detektor: {detector_backend}): {type(e)._name_} - {str(e)}"
        return {"face_bytes": None, "original_region": None, "error": error_msg}
    finally:
        if img_path_original and os.path.exists(img_path_original):
            os.remove(img_path_original)