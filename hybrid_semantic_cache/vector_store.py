import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Inisialisasi FAISS Index.
        Dimension 384 sesuai dengan output model 'MiniLM-L12-v2'.
        """
        # Kita gunakan IndexFlatIP (Inner Product) untuk menghitung Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        # FAISS hanya menyimpan angka, jadi kita butuh list tambahan 
        # untuk menyimpan teks asli dan jawabannya (Metadata)
        self.metadata = []

    def add_to_index(self, question_vectors: np.ndarray, qa_pairs: list):
        """
        Menambahkan kumpulan vektor ke dalam lemari FAISS.
        QA_pairs berisi list dict {'question': ..., 'answer': ...}
        """
        # Normalisasi vektor agar perhitungan Inner Product menjadi Cosine Similarity
        faiss.normalize_L2(question_vectors)
        
        # Masukkan ke FAISS
        self.index.add(question_vectors)
        
        # Simpan teks aslinya agar saat 'Hit', kita tahu jawaban mana yang harus diambil
        self.metadata.extend(qa_pairs)
        print(f"[VECTOR STORE] Berhasil memuat {len(qa_pairs)} data ke memori.")

    def search(self, query_vector: np.ndarray, top_k: int = 1):
        """
        Mencari pertanyaan yang paling mirip di dalam lemari.
        Return: (score, metadata_item)
        """
        # Normalisasi query vector
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # D adalah Score (Similarity), I adalah Index (posisi di metadata)
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        
        score = float(distances[0][0])
        idx = int(indices[0][0])
        
        # Jika indeks -1 berarti tidak ditemukan apa pun di FAISS
        if idx != -1:
            return score, self.metadata[idx]
        return 0.0, None

# --- SINGLETON INSTANCE ---
# Lemari ini harus satu saja agar datanya tidak terpencar
vector_store = VectorStore()