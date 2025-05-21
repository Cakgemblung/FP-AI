import streamlit as st
from PIL import Image
import io
from src.face_processing.core import verify_images # Memuat fungsi verifikasi wajah

# --- Fungsi untuk memuat CSS kustom (opsional) ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Abaikan jika file CSS tidak ditemukan

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Uji Similaritas Wajah",
    page_icon="🎭",
    layout="wide"
)

# local_css("assets/css/custom_style.css") # Aktifkan jika menggunakan CSS kustom

# --- Inisialisasi Session State untuk data aplikasi ---
if 'img1_bytes' not in st.session_state:
    st.session_state.img1_bytes = None
if 'img2_bytes' not in st.session_state:
    st.session_state.img2_bytes = None
if 'img1_name' not in st.session_state:
    st.session_state.img1_name = None
if 'img2_name' not in st.session_state:
    st.session_state.img2_name = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --- Tampilan Utama Aplikasi ---
st.title("🎭 Website Uji Similaritas Wajah")
st.markdown("Unggah dua gambar wajah untuk memulai analisis similaritas.")

# --- Kontrol di Sidebar untuk unggah gambar dan prediksi ---
with st.sidebar:
    st.header("🖼️ Unggah Gambar")
    uploaded_file_1 = st.file_uploader("Pilih Gambar Wajah 1", type=["jpg", "jpeg", "png"], key="uploader1")
    if uploaded_file_1 is not None:
        st.session_state.img1_bytes = uploaded_file_1.getvalue()
        st.session_state.img1_name = uploaded_file_1.name
        st.session_state.prediction_result = None # Reset hasil jika gambar 1 diubah

    uploaded_file_2 = st.file_uploader("Pilih Gambar Wajah 2", type=["jpg", "jpeg", "png"], key="uploader2")
    if uploaded_file_2 is not None:
        st.session_state.img2_bytes = uploaded_file_2.getvalue()
        st.session_state.img2_name = uploaded_file_2.name
        st.session_state.prediction_result = None # Reset hasil jika gambar 2 diubah

    st.markdown("---")
    predict_button = st.button("🔎 Prediksi Similaritas", type="primary", use_container_width=True, key="predict")
    st.markdown("---")
    st.caption("Integrasi DeepFace")


# --- Area Tampilan Gambar (dalam dua kolom) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gambar 1")
    if st.session_state.img1_bytes:
        try:
            image1 = Image.open(io.BytesIO(st.session_state.img1_bytes))
            # Menggunakan use_container_width untuk responsivitas dan menghilangkan warning
            st.image(image1, caption=f"{st.session_state.img1_name}", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat Gambar 1: {str(e)}") # Menampilkan pesan error spesifik
            st.session_state.img1_bytes = None # Reset jika error
            st.session_state.img1_name = None
    else:
        st.info("Silakan unggah Gambar 1 melalui sidebar.")

with col2:
    st.subheader("Gambar 2")
    if st.session_state.img2_bytes:
        try:
            image2 = Image.open(io.BytesIO(st.session_state.img2_bytes))
            # Menggunakan use_container_width untuk responsivitas dan menghilangkan warning
            st.image(image2, caption=f"{st.session_state.img2_name}", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat Gambar 2: {str(e)}") # Menampilkan pesan error spesifik
            st.session_state.img2_bytes = None # Reset jika error
            st.session_state.img2_name = None
    else:
        st.info("Silakan unggah Gambar 2 melalui sidebar.")

# --- Logika Tombol Prediksi & Area Tampilan Hasil ---
st.markdown("---")
st.subheader("Hasil Prediksi")
result_area = st.container() # Kontainer untuk hasil agar terorganisir

# Logika ketika tombol prediksi ditekan
if predict_button:
    if st.session_state.img1_bytes and st.session_state.img2_bytes:
        with st.spinner("Menganalisis similaritas wajah... Ini mungkin memakan waktu."):
            # Memanggil fungsi verifikasi dari modul core
            result = verify_images(st.session_state.img1_bytes, st.session_state.img2_bytes)
            st.session_state.prediction_result = result # Simpan hasil ke session state
    elif st.session_state.img1_bytes and not st.session_state.img2_bytes:
        st.session_state.prediction_result = {"error": "Mohon unggah Gambar 2 untuk melanjutkan."}
    elif not st.session_state.img1_bytes and st.session_state.img2_bytes:
        st.session_state.prediction_result = {"error": "Mohon unggah Gambar 1 untuk melanjutkan."}
    else:
        st.session_state.prediction_result = {"error": "Mohon unggah Gambar 1 dan Gambar 2 untuk melanjutkan."}

# Menampilkan hasil prediksi dari session state
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    with result_area:
        if "error" in result and result["error"] is not None: # Periksa apakah ada pesan error
            st.error(f"Terjadi Kesalahan: {result['error']}")
        elif result and "verified" in result: # Pastikan 'result' tidak None dan memiliki kunci 'verified'
            verified = result.get("verified", False)
            distance = result.get("distance", 0.0)
            threshold = result.get("threshold", 0.0)
            model_name = result.get("model", "N/A")
            similarity_metric = result.get("similarity_metric", "N/A")

            if verified:
                st.success("✅ Wajah Terverifikasi Mirip!")
            else:
                st.warning("❌ Wajah Tidak Terverifikasi Mirip.")

            st.markdown(f"""
            - **Distance:** `{distance:.4f}` (Semakin kecil, semakin mirip)
            - **Threshold:** `{threshold:.4f}` (Batas untuk verifikasi model ini)
            - **Model:** `{model_name}`
            - **Similarity Metric:** `{similarity_metric}`
            """)
        else:
            # Kasus jika result tidak memiliki struktur yang diharapkan atau None
            st.info("Hasil prediksi tidak tersedia atau format tidak dikenali.")
else:
     with result_area:
        st.write("Hasil analisis similaritas akan ditampilkan di sini setelah Anda menekan tombol 'Prediksi'.")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 - Tugas Final Project AI | Menggunakan DeepFace")