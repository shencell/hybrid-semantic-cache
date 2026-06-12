"""Indonesian slang / typo normaliser.

Rewrites colloquial chat language ("gmn cr ganti pw yak?") into standard
Indonesian before embedding, which substantially raises semantic-cache hit
rates. Purely lexical - fast, deterministic, and free of any LLM call.
"""

from __future__ import annotations

import re


class TextNormalizer:
    def __init__(self):
            self.slang_dict = {
                # --- TRANSLASI KATA KERJA & SIFAT ---
                "gmn": "bagaimana", "gimana": "bagaimana", "cr": "cara", "caranya": "cara",
                "ganti": "mengubah", "rubah": "mengubah", "edit": "mengubah", "benerinnya": "memperbaiki",
                "aktifin": "mengaktifkan", "hapus": "menghapus", "ilangin": "menyembunyikan",
                "masukin": "memasukkan", "gabungin": "menautkan", "dipindah": "memindahkan",
                "ngecilinnya": "mengecilkan", "ngasih": "memberikan", "matiin": "mematikan",
                "amanin": "mengamankan", "ilang": "hilang", "nemu": "menemukan", "laporin": "melaporkan",
                "batalin": "membatalkan", "narik": "menarik", "nyimpen": "menyimpan", "cetak": "mencetak",
                "hubungin": "menghubungi", "downloadnya": "mengunduh",
                "kedaftar": "terdaftar", "keblokir": "terblokir", "kesimpen": "tersimpan",
                "kegedean": "terlalu besar", "lemot": "lambat", "kenceng": "cepat",

                # --- TRANSLASI BENDA & ISTILAH IT ---
                "pw": "kata sandi", "password": "kata sandi", "pass": "kata sandi", "pin": "sandi",
                "nomer": "nomor", "hp": "perangkat", "ponsel": "perangkat", "apk": "aplikasi",
                "app": "aplikasi", "aplikasinya": "aplikasi", "notif": "notifikasi", "refferal": "referral",
                "ig": "instagram", "sosmed": "media sosial", "indo": "indonesia", "cc": "kartu kredit",
                "duit": "dana", "saldo": "dana", "tf": "transfer", "invoice": "struk",
                "pesenan": "pesanan", "loker": "lowongan pekerjaan", "prusahaan": "perusahaan",
                "cs": "layanan pelanggan", "bot": "sistem", "pc": "komputer", "laptop": "komputer",
                "eror": "gangguan", "ngebug": "gangguan", "bug": "celah keamanan", "maintenance": "pemeliharaan",

                # --- TRANSLASI KATA SAMBUNG & KETERANGAN ---
                "klo": "kalau", "kalo": "kalau", "gk": "tidak", "gak": "tidak", "nggak": "tidak", "ga": "tidak",
                "bisa": "dapat", "bs": "dapat", "udah": "sudah", "udh": "sudah", "blm": "belum",
                "trs": "terus", "mulu": "selalu", "pdhl": "padahal", "kpn": "kapan", "dmn": "dimana",
                "kmn": "kemana", "dr": "dari", "drtd": "dari tadi", "brp": "berapa", "karna": "karena",
                "gara2": "karena", "tp": "tetapi", "bgt": "sangat", "buat": "untuk", "bwt": "untuk",
                "lg": "lagi", "sampe": "sampai", "abis": "setelah", "lwt": "melalui", "mn": "mana",
                "sm": "dengan", "ato": "atau", "pas": "saat",

                # --- PENGHAPUSAN STOPWORDS (KATA SAMPAH) ---
                "min": "", "mimin": "", "admin": "", "yak": "", "ya": "", "yg": "",
                "dong": "", "sih": "", "nih": "", "eh": "", "doang": "", "bro": "",
                "bang": "", "gw": "", "gua": "", "aku": "", "lu": "", "lo": "",
                "biar": "", "kok": "", "apaan": "", "tuh": ""
            }

    def normalize(self, text: str) -> str:
        """
        Fungsi ini akan membersihkan dan membakukan teks sebelum dikirim ke AI
        """
        # 1. Ubah semua huruf menjadi kecil (Case Folding)
        text = text.lower()

        # 2. Hilangkan tanda baca yang tidak perlu
        text = re.sub(r'[^\w\s]', '', text)

        # 3. Pecah kalimat menjadi kata per kata
        words = text.split()

        # 4. Ganti kata gaul atau hapus kata sampah
        normalized_words = []
        for word in words:
            if word in self.slang_dict:
                translated = self.slang_dict[word]
                # Hanya masukkan ke kalimat final jika kata terjemahannya tidak kosong
                if translated != "":
                    normalized_words.append(translated)
            else:
                normalized_words.append(word)

        # 5. Gabungkan kembali menjadi satu kalimat utuh
        final_text = " ".join(normalized_words)

        return final_text


# Singleton instance
normalizer = TextNormalizer()


def normalize_text(text: str) -> str:
    """Module-level convenience wrapper around the shared :class:`TextNormalizer`.

    This is the function documented in the README/Quick Start::

        from hybrid_semantic_cache import normalize_text
        clean = normalize_text("Tlong bkinin srt resign dong")
    """
    return normalizer.normalize(text)
