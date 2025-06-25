import streamlit as st
from PIL import Image, ImageDraw
import io
from src.face_processing.core import verify_images, analyze_face_attributes, extract_aligned_face_bytes

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="✨ Analisis Wajah Pro ✨",
    page_icon="🤖",
    layout="wide", # Tetap wide untuk memanfaatkan warna body, konten utama akan di-center
    initial_sidebar_state="expanded"
)

# --- Fungsi untuk CSS Kustom ---
def load_custom_css():
    st.markdown("""
    <style>
        /* Latar Belakang Utama Aplikasi */
        body {
            background-color: #E0F7FA; /* Warna biru pastel muda */
        }
        
        /* Kontainer utama Streamlit */
        .main .block-container { 
            padding-top: 1.5rem; padding-bottom: 2rem; /* Kurangi padding atas sedikit */
            padding-left: 1.5rem; padding-right: 1.5rem;
            background-color: #FFFFFF; 
            border-radius: 15px; 
            box-shadow: 0 8px 16px rgba(0,0,0,0.1); 
            margin: 1.5rem auto; /* Center block-container jika layout="wide" */
            max-width: 1100px; /* Batasi lebar kontainer utama agar tidak terlalu stretched */
        }

        /* Sidebar */
        [data-testid="stSidebar"] > div:first-child {
            background-color: #B2EBF2; 
            padding: 20px; 
            border-radius: 0 10px 10px 0;
        }
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys p {
            color: #004D40; 
            font-weight: 500;
        }

        /* Gambar dan Caption */
        .stImage > img {
            max-width: 280px !important; height: auto !important;    
            display: block; margin-left: auto; margin-right: auto;
            border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 2px solid #B2EBF2; 
        }
        .stImage > figcaption {
            font-style: italic; font-size: 0.85em; text-align: center;
            margin-top: 8px; color: #424242; 
        }

        /* Expander Atribut */
        .streamlit-expanderHeader {
            font-size: 1em; font-weight: bold; color: #004D40; 
            background-color: #E0F2F1; 
            border-radius: 7px; padding: 10px 15px !important;
        }
        .streamlit-expanderContent {
            background-color: #FAFAFA; border: 1px solid #E0F2F1; border-top: none;
            border-radius: 0 0 7px 7px; padding: 15px;
        }

        /* Tombol Analisis Utama */
        div.stButton > button[kind="primary"] {
            background-color: #00796B; color: white; font-weight: bold; border-radius: 25px;
            padding: 12px 25px; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease; 
        }
        div.stButton > button[kind="primary"]:hover { background-color: #004D40; }
        
        /* Pesan Sukses, Warning, Error */
        .success-message { padding: 12px; background-color: #E8F5E9; color: #2E7D32; border-left: 5px solid #4CAF50; border-radius: 5px; margin-bottom: 10px; }
        .warning-message { padding: 12px; background-color: #FFFDE7; color: #F57F17; border-left: 5px solid #FFC107; border-radius: 5px; margin-bottom: 10px; }
        .error-message-custom { padding: 12px; background-color: #FFEBEE; color: #C62828; border-left: 5px solid #D32F2F; border-radius: 5px; margin-bottom: 10px; }
        
        /* Garis Pemisah HR */
        hr.styled-hr { border: 0; height: 1.5px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 77, 64, 0.25), rgba(0, 0, 0, 0)); margin: 2em 0; }

        /* Judul Utama dan Deskripsi Aplikasi */
        .app-title { text-align: center; color: #004D40; margin-bottom: 0.3em; font-weight: 600; }
        .app-description-centered { /* Class BARU untuk deskripsi */
            text-align: center !important;
            font-size: 1.1em; 
            color: #00695C; 
            max-width: 700px; /* Sesuaikan max-width jika perlu */
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 1.5em; /* Jarak bawah */
        }
        .section-title { color: #004D40; margin-top: 1.5em; margin-bottom: 0.8em; border-bottom: 2px solid #B2EBF2; padding-bottom: 0.3em; }
        .image-pair-title { color: #00695C; font-size: 1.2em; margin-bottom:0.5em; text-align:center; }

    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --- Inisialisasi Session State (Sama) ---
SESSION_KEYS_DEFAULTS = {
    'img1_bytes_original': None, 'img2_bytes_original': None, 'img1_name': None, 'img2_name': None,
    'img1_cropped_bytes': None, 'img2_cropped_bytes': None, 'img1_original_region': None, 'img2_original_region': None,
    'similarity_result': None, 'img1_attributes': None, 'img2_attributes': None,
    'selected_model': "VGG-Face", 'selected_detector': "opencv", 'selected_distance_metric': "cosine",
    'analysis_button_clicked': False
}
for key, default_value in SESSION_KEYS_DEFAULTS.items():
    if key not in st.session_state: st.session_state[key] = default_value


# --- Tampilan Utama Aplikasi ---
st.markdown("<h1 class='app-title'>🎭 Analisis Wajah Komprehensif 🖼</h1>", unsafe_allow_html=True)
# Gunakan class CSS baru untuk deskripsi
st.markdown("<p class='app-description-centered'>Unggah gambar wajah untuk analisis mendalam: umur, emosi, gender, ras, dan uji similaritas dengan berbagai konfigurasi.</p>", unsafe_allow_html=True)
st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)


# --- Fungsi Utilitas Reset (Sama) ---
def reset_specific_image_states(img_key_prefix):
    keys_to_reset = [f'{img_key_prefix}_bytes_original', f'{img_key_prefix}_name', 
                     f'{img_key_prefix}_cropped_bytes', f'{img_key_prefix}_original_region', 
                     f'{img_key_prefix}_attributes', 'similarity_result', 'analysis_button_clicked']
    for key in keys_to_reset:
        if key in st.session_state: st.session_state[key] = SESSION_KEYS_DEFAULTS.get(key)

def reset_all_on_setting_change():
    keys_to_reset = [k for k in SESSION_KEYS_DEFAULTS if k not in ['selected_model', 'selected_detector', 'selected_distance_metric']]
    for key in keys_to_reset:
        st.session_state[key] = SESSION_KEYS_DEFAULTS.get(key)


# --- Kontrol di Sidebar (Sama) ---
with st.sidebar:
    st.markdown("## ⚙ Pengaturan Analisis")
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    distance_metrics = ["cosine", "euclidean", "euclidean_l2"]
    st.selectbox("Model Similaritas:", models, key="selected_model", on_change=reset_all_on_setting_change)
    st.selectbox("Detektor Wajah:", detectors, key="selected_detector", on_change=reset_all_on_setting_change)
    st.selectbox("Metrik Jarak (Similaritas):", distance_metrics, key="selected_distance_metric", on_change=reset_all_on_setting_change)
    st.markdown("---")
    st.markdown("## 🖼 Unggah Gambar Asli")
    def handle_file_upload(img_prefix, uploader_key):
        reset_specific_image_states(img_prefix)
        uploaded_file = st.session_state[uploader_key]
        if uploaded_file is not None:
            st.session_state[f'{img_prefix}_bytes_original'] = uploaded_file.getvalue()
            st.session_state[f'{img_prefix}_name'] = uploaded_file.name
            with st.spinner(f"Mengekstrak wajah Gbr {img_prefix[-1]}..."):
                extract_res = extract_aligned_face_bytes(st.session_state[f'{img_prefix}_bytes_original'], st.session_state.selected_detector)
                if extract_res and not extract_res.get("error"):
                    st.session_state[f'{img_prefix}_cropped_bytes'] = extract_res["face_bytes"]
                    st.session_state[f'{img_prefix}_original_region'] = extract_res["original_region"]
                else: 
                    st.sidebar.error(f"Gbr {img_prefix[-1]}: {extract_res.get('error', 'Gagal extract wajah.')}")
                    st.session_state[f'{img_prefix}_cropped_bytes'] = None 
                    st.session_state[f'{img_prefix}_original_region'] = None
    st.file_uploader("Pilih Gambar Wajah 1", type=["jpg", "jpeg", "png"], key="uploader_img1", on_change=handle_file_upload, args=("img1", "uploader_img1"))
    st.file_uploader("Pilih Gambar Wajah 2", type=["jpg", "jpeg", "png"], key="uploader_img2", on_change=handle_file_upload, args=("img2", "uploader_img2"))
    st.markdown("---")
    analyze_button_disabled = not (st.session_state.img1_cropped_bytes or st.session_state.img2_cropped_bytes)
    analyze_button = st.button("🚀 Analisis & Prediksi Sekarang!", type="primary", use_container_width=True, disabled=analyze_button_disabled)
    if analyze_button: st.session_state.analysis_button_clicked = True
    st.markdown("---")
    st.info(f"Konf: {st.session_state.selected_model}, *{st.session_state.selected_detector}, {st.session_state.selected_distance_metric}*.")

# --- Logika Tombol Analisis (Sama) ---
if st.session_state.analysis_button_clicked:
    st.session_state.similarity_result = None; st.session_state.img1_attributes = None; st.session_state.img2_attributes = None
    can_analyze_img1 = st.session_state.img1_cropped_bytes is not None; can_analyze_img2 = st.session_state.img2_cropped_bytes is not None
    if not can_analyze_img1 and not can_analyze_img2: st.warning("Tidak ada wajah yang berhasil di-extract untuk dianalisis.")
    else:
        with st.spinner(f"Menganalisis..."):
            model, detector, metric = st.session_state.selected_model, st.session_state.selected_detector, st.session_state.selected_distance_metric
            if can_analyze_img1: st.session_state.img1_attributes = analyze_face_attributes(st.session_state.img1_cropped_bytes, detector)
            if can_analyze_img2: st.session_state.img2_attributes = analyze_face_attributes(st.session_state.img2_cropped_bytes, detector)
            if can_analyze_img1 and can_analyze_img2: st.session_state.similarity_result = verify_images(st.session_state.img1_cropped_bytes, st.session_state.img2_cropped_bytes, model, detector, metric)

# --- Fungsi untuk menampilkan atribut wajah (Sama) ---
def display_attributes_section(attributes_data, image_number_str):
    if attributes_data:
        if attributes_data.get("error"): st.markdown(f"<div class='error-message-custom'>Atribut Gbr {image_number_str}: {attributes_data['error']}</div>", unsafe_allow_html=True)
        elif attributes_data.get("data") and len(attributes_data["data"]) > 0:
            face_data = attributes_data["data"][0]
            age, emotion, gender, race = face_data.get('age','N/A'), face_data.get('dominant_emotion','N/A').capitalize(), face_data.get('dominant_gender','N/A').capitalize(), face_data.get('dominant_race','N/A').capitalize()
            st.markdown(f"Umur: {age} | Emosi: {emotion}")
            st.markdown(f"Gender: {gender} | Ras: {race}")
            if len(attributes_data["data"]) > 1: st.caption(f"(Info untuk wajah pertama dari {len(attributes_data['data'])} wajah)")
        else: st.info(f"Atribut Gbr {image_number_str}: Data tidak valid atau wajah tidak terdeteksi.")

# --- Area Tampilan Gambar & Atribut (Penyesuaian kolom dan judul) ---
st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>🔬 Tampilan Gambar & Atribut Wajah</h2>", unsafe_allow_html=True)

main_cols = st.columns(2, gap="large") 

with main_cols[0]:
    st.markdown(f"<h3 class='image-pair-title'>👤 Gambar 1: {st.session_state.img1_name or 'Belum Diunggah'}</h3>", unsafe_allow_html=True)
    if st.session_state.img1_bytes_original:
        # Tampilkan Asli dan Crop dalam sub-kolom jika keduanya ada, atau satu per satu jika hanya asli
        if st.session_state.img1_cropped_bytes:
            sub_cols1 = st.columns(2)
            with sub_cols1[0]:
                st.markdown("<h6>Gambar Asli</h6>", unsafe_allow_html=True)
                img_pil = Image.open(io.BytesIO(st.session_state.img1_bytes_original))
                if st.session_state.img1_original_region:
                    draw = ImageDraw.Draw(img_pil); r = st.session_state.img1_original_region
                    draw.rectangle([r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']], outline="#66BB6A", width=5)
                st.image(img_pil, use_container_width=True) 
            with sub_cols1[1]:
                st.markdown("<h6>Wajah (Crop & Align)</h6>", unsafe_allow_html=True)
                st.image(st.session_state.img1_cropped_bytes, use_container_width=True)
        else: # Hanya tampilkan gambar asli jika crop gagal atau belum ada
            st.markdown("<h6>Gambar Asli</h6>", unsafe_allow_html=True)
            img_pil_orig_only = Image.open(io.BytesIO(st.session_state.img1_bytes_original))
            st.image(img_pil_orig_only, use_container_width=True)
            if st.session_state.analysis_button_clicked: st.warning("Ekstraksi wajah untuk Gambar 1 gagal.")

        if st.session_state.img1_attributes and st.session_state.img1_cropped_bytes: 
            with st.expander("🔍 Lihat Atribut Gambar 1", expanded=True): 
                display_attributes_section(st.session_state.img1_attributes, "1")
    else: 
        st.info("Unggah Gambar 1 di sidebar.")


with main_cols[1]: 
    st.markdown(f"<h3 class='image-pair-title'>👤 Gambar 2: {st.session_state.img2_name or 'Belum Diunggah'}</h3>", unsafe_allow_html=True)
    if st.session_state.img2_bytes_original:
        if st.session_state.img2_cropped_bytes:
            sub_cols2 = st.columns(2)
            with sub_cols2[0]:
                st.markdown("<h6>Gambar Asli</h6>", unsafe_allow_html=True)
                img_pil = Image.open(io.BytesIO(st.session_state.img2_bytes_original))
                if st.session_state.img2_original_region:
                    draw = ImageDraw.Draw(img_pil); r = st.session_state.img2_original_region
                    draw.rectangle([r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']], outline="#66BB6A", width=5)
                st.image(img_pil, use_container_width=True)
            with sub_cols2[1]:
                st.markdown("<h6>Wajah (Crop & Align)</h6>", unsafe_allow_html=True)
                st.image(st.session_state.img2_cropped_bytes, use_container_width=True)
        else: # Hanya tampilkan gambar asli jika crop gagal
            st.markdown("<h6>Gambar Asli</h6>", unsafe_allow_html=True)
            img_pil_orig_only_2 = Image.open(io.BytesIO(st.session_state.img2_bytes_original))
            st.image(img_pil_orig_only_2, use_container_width=True)
            if st.session_state.analysis_button_clicked: st.warning("Ekstraksi wajah untuk Gambar 2 gagal.")

        if st.session_state.img2_attributes and st.session_state.img2_cropped_bytes:
            with st.expander("🔍 Lihat Atribut Gambar 2", expanded=True): 
                display_attributes_section(st.session_state.img2_attributes, "2")
    else: 
        st.info("Unggah Gambar 2 di sidebar.")


# --- Area Tampilan Hasil Similaritas (Sama) ---
# ... (Kode ini tidak berubah signifikan dari versi sebelumnya)
if st.session_state.img1_cropped_bytes and st.session_state.img2_cropped_bytes:
    st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>⚖ Hasil Analisis Similaritas</h2>", unsafe_allow_html=True)
    results_placeholder_bottom = st.container()
    with results_placeholder_bottom:
        if st.session_state.similarity_result:
            result_sim = st.session_state.similarity_result
            if "error" in result_sim and result_sim["error"] is not None: st.markdown(f"<div class='error-message-custom'>{result_sim['error']}</div>", unsafe_allow_html=True)
            elif "verified" in result_sim: 
                verified, dist, thres = result_sim.get("verified", False), result_sim.get("distance", 0.0), result_sim.get("threshold", 0.0)
                model, detector, metric = result_sim.get("model_name_used", "N/A"), result_sim.get("detector_backend_used", "N/A"), result_sim.get("distance_metric_used", "N/A")
                if verified: st.markdown(f"<div class='success-message'>✅ Wajah Terverifikasi Mirip! (Distance: {dist:.4f} ≤ Threshold: {thres:.2f})</div>", unsafe_allow_html=True)
                else: st.markdown(f"<div class='warning-message'>❌ Tidak Mirip. (Distance: {dist:.4f} > Threshold: {thres:.2f})</div>", unsafe_allow_html=True)
                with st.expander("Detail Konfigurasi Similaritas"):
                    st.markdown(f"- Model: {model}\n- Detektor Awal: {st.session_state.selected_detector}\n- Metrik: {metric}")
        elif st.session_state.analysis_button_clicked: st.info("Proses similaritas belum menghasilkan data atau gagal.")
elif st.session_state.analysis_button_clicked and (st.session_state.img1_bytes_original or st.session_state.img2_bytes_original):
    st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>⚖ Hasil Analisis Similaritas</h2>", unsafe_allow_html=True)
    st.info("Analisis similaritas memerlukan dua wajah yang berhasil di-crop.")
if not st.session_state.analysis_button_clicked and not st.session_state.img1_bytes_original and not st.session_state.img2_bytes_original:
    st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
    st.info("👋 Selamat datang! Silakan unggah gambar dan pilih pengaturan, lalu klik tombol analisis.")

# --- Footer (Sama) ---
st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #757575; font-size: 0.9em;'>© 2025 - Proyek Analisis Wajah</p>", unsafe_allow_html=True)
