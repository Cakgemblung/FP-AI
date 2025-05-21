from deepface import DeepFace
import tempfile
import os

# Konfigurasi model dan metrik (bisa disesuaikan)
MODEL_NAME = "VGG-Face" # Pilihan: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
DETECTOR_BACKEND = "opencv" # Pilihan: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
DISTANCE_METRIC = "cosine" # Pilihan: "cosine", "euclidean", "euclidean_l2"

def verify_images(img1_bytes, img2_bytes):
    img1_path = None
    img2_path = None
    results = None

    try:
        # DeepFace.verify memerlukan path file, jadi kita buat file temporer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img1:
            tmp_img1.write(img1_bytes)
            img1_path = tmp_img1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img2:
            tmp_img2.write(img2_bytes)
            img2_path = tmp_img2.name

        # Lakukan verifikasi
        # enforce_detection=True akan error jika wajah tidak terdeteksi
        # enforce_detection=False akan mencoba membandingkan gambar walau tanpa wajah (kurang akurat untuk wajah)
        results = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True # Penting: pastikan wajah terdeteksi
        )
        return results

    except ValueError as ve: # Sering terjadi jika wajah tidak terdeteksi
        # print(f"ValueError during DeepFace processing: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        # print(f"An unexpected error occurred during DeepFace processing: {e}")
        return {"error": f"Terjadi kesalahan: {str(e)}"}
    finally:
        # Hapus file temporer
        if img1_path and os.path.exists(img1_path):
            os.remove(img1_path)
        if img2_path and os.path.exists(img2_path):
            os.remove(img2_path)