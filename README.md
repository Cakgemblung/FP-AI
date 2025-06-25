# Website Uji Similaritas Wajah

Aplikasi web untuk menguji similaritas antara dua gambar wajah menggunakan Streamlit dan DeepFace.

## Prasyarat

*   Python 3.10+ (misalnya, Python 3.12)
*   `pip` (Python package installer)
*   (Windows) [Microsoft Visual C++ Redistributable for Visual Studio 2015-2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) (x64, restart setelah instalasi).

## Setup dan Menjalankan Aplikasi

### 1. Clone Repositori (Jika Belum)
```bash
git clone [URL_GITHUB_ANDA_DI_SINI]
cd [NAMA_FOLDER_PROYEK_ANDA]
```

### 2. Setup Virtual Environment
Sangat direkomendasikan untuk menggunakan virtual environment.

*   **Buat Virtual Environment:**
    (Di dalam direktori root proyek)
    ```bash
    python -m venv venv_app
    ```
    *(Anda bisa mengganti `venv_app` dengan nama lain)*

*   **Aktifkan Virtual Environment:**
    *   Windows (Command Prompt/PowerShell):
        ```bash
        venv_app\Scripts\activate
        ```
    *   macOS/Linux (Bash/Zsh):
        ```bash
        source venv_app/bin/activate
        ```
    *(Prompt terminal akan menampilkan `(venv_app)` di depannya)*

### 3. Instal Dependensi
Pastikan virtual environment (`venv_app`) sudah aktif.
```bash
pip install -r requirements.txt
```
*(Proses ini mungkin memerlukan waktu. DeepFace juga akan mengunduh model saat pertama kali digunakan di aplikasi).*

### 4. Jalankan Aplikasi Streamlit
Pastikan virtual environment (`venv_app`) masih aktif.
```bash
streamlit run app.py
```
Aplikasi akan tersedia di `http://localhost:8501` di browser Anda.

### 5. Deaktivasi Virtual Environment
```bash
deactivate
```
