from fastapi import FastAPI
import time
import json
import os
from api.routes import router as chat_router
from core.embedding import embedder
from core.vector_store import vector_store

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Semantic Caching Middleware",
    description="Eksperimen Jaringan untuk Skripsi: Evaluasi Prompt Gaul Indonesia",
    version="1.0.0"
)

# --- FASE PRE-WARMING ---
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("🚀 [STARTUP] Menyalakan Semantic Caching Middleware...")
    start_time = time.perf_counter()
    
    try:
        # 1. Baca file JSON baku
        file_path = os.path.join(os.path.dirname(__file__), "data", "dataset_baku.json")
        with open(file_path, "r", encoding="utf-8") as file:
            qa_pairs = json.load(file)
        
        # 2. Ekstrak hanya teks pertanyaannya saja untuk diubah jadi vektor
        questions = [item["question"] for item in qa_pairs]
        
        print(f"📥 [STARTUP] Membaca {len(questions)} data dari JSON...")
        
        # 3. Ubah semua pertanyaan baku menjadi Vektor (Batch Processing)
        question_vectors = embedder.encode_batch(questions)
        
        # 4. Masukkan vektor dan jawaban lengkapnya ke lemari FAISS
        vector_store.add_to_index(question_vectors, qa_pairs)
        
        end_time = time.perf_counter()
        print(f"✅ [STARTUP] Pre-Warming sukses dalam {end_time - start_time:.4f} detik.")
        print("📦 [STATUS] Gudang FAISS siap menerima traffic AI!")
    
    except Exception as e:
        print(f"❌ [STARTUP ERROR] Gagal memuat data: {e}")
        
    print("="*50 + "\n")

# Daftarkan gerbang API
app.include_router(chat_router)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "Middleware Semantic Cache Berjalan. Tembak prompt ke /chat"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)