from sentence_transformers import SentenceTransformer
import numpy as np
import time

class TextEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inisialisasi model AI saat class pertama kali dipanggil.
        Proses ini hanya terjadi satu kali untuk menghemat RAM dan CPU.
        """
        print(f"\n[EMBEDDING] Sedang mengunduh/memuat model '{model_name}'...")
        start_time = time.perf_counter()
        
        # HuggingFace otomatis akan mengunduh model jika belum ada, 
        # dan memuat dari cache lokal jika sudah pernah diunduh.
        self.model = SentenceTransformer(model_name)
        
        end_time = time.perf_counter()
        print(f"[EMBEDDING] Model berhasil dimuat dalam {end_time - start_time:.2f} detik.")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Mengubah 1 kalimat (prompt masuk) menjadi vektor.
        Digunakan pada Fase Pengujian (saat traffic gaul masuk).
        """
        # Encode mengembalikan list, kita pastikan formatnya numpy array float32
        # karena FAISS hanya menerima format tipe data float32
        vector = self.model.encode(text)
        return np.array(vector).astype('float32')

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Mengubah banyak kalimat sekaligus menjadi kumpulan vektor.
        Digunakan khusus pada Fase Pre-Warming (memuat dataset baku).
        """
        vectors = self.model.encode(texts)
        return np.array(vectors).astype('float32')

# --- SINGLETON INSTANCE ---
# Kita membuat instance global agar model tidak dimuat ulang berkali-kali
# setiap ada request API masuk. Ini kunci agar latency tetap rendah!
embedder = TextEmbedder()