========================================================================
SOURCE CODE SKRIPSI: HYBRID SEMANTIC CACHE MIDDLEWARE
Oleh: Sandy Wicaksono
========================================================================

A. DESKRIPSI PROYEK
Proyek ini adalah implementasi arsitektur Hybrid Semantic Cache Middleware menggunakan model embedding all-MiniLM-L6-v2 dan database vektor FAISS. Middleware ini bertujuan untuk mencegat kueri bahasa gaul/typo dari pengguna (melalui modul Normalizer) dan membalasnya secara lokal untuk memotong latensi (waktu respons) dari LLM Google Gemini 3.1 Flash Lite di Cloud.

B. PRASYARAT (PREREQUISITES)
1. Python versi 3.10 atau 3.11 telah terinstal.
2. Memiliki API Key dari Google Gemini (Google AI Studio).

C. CARA INSTALASI
1. Buka Terminal / Command Prompt di dalam folder proyek ini.
2. (Opsional) Buat virtual environment: 
   python -m venv env_skripsi
   Lalu aktifkan:
   - Windows: .\env_skripsi\Scripts\activate
   - Mac/Linux: source env_skripsi/bin/activate
3. Instal semua library yang dibutuhkan dengan perintah:
   pip install -r requirements.txt

D. PENGATURAN API KEY GEMINI
Buka file `src/gemini_api.py` dan masukkan API Key Gemini Anda pada variabel:
API_KEY = "MASUKKAN_API_KEY_ANDA_DI_SINI"

E. CARA MENJALANKAN SERVER MIDDLEWARE (FASTAPI)
1. Buka terminal di direktori utama proyek.
2. Jalankan perintah berikut:
   uvicorn src.main:app --reload
3. Server lokal akan berjalan di http://127.0.0.1:8000
4. Anda dapat mencoba endpoint dengan membuka URL Swagger UI di:
   http://127.0.0.1:8000/docs

F. CARA MENJALANKAN PENGUJIAN (BAB 4 SKRIPSI)
Untuk menjalankan simulasi pengujian Skenario A, B, dan C (Hit Rate & Latency):
1. Buka terminal baru.
2. Jalankan script pengujian:
   python evaluasi/test_scenarios.py
3. Untuk membuat grafik visualisasi hasil pengujian, jalankan:
   python evaluasi/generate_grafik.py
4. File gambar grafik (.png) akan otomatis tersimpan di dalam folder `output/`.

========================================================================
Terima kasih. Jika ada kendala dalam menjalankan program, silakan 
menghubungi peneliti (Sandy).
========================================================================