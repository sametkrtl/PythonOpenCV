import os
import fitz  # PyMuPDF
import requests
import json
from pymongo import MongoClient

# =============================== #
# CONFIG
# =============================== #
PDF_PATH = "C:/Users/stj.skartal/Desktop/samet/pdfs/ornek_veri2.pdf"  # Buraya kendi PDF yolunu yaz
OPENROUTER_API_KEY = os.getenv("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")  # .env dosyasında tanımlı olmalı
MODEL_ID = "meta-llama/llama-3.2-11b-vision-instruct:free"

# MongoDB Bağlantısı
client = MongoClient("mongodb://localhost:27017/")
db = client["cv_veritabani"]
collection = db["cv_kayitlari"]

# =============================== #
# PDF'ten Metin Çıkar
# =============================== #
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# =============================== #
# LLM'e Soru Sor
# =============================== #
def ask_qwen_openrouter(prompt: str, api_key: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "cv-parser"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(url, headers=headers, json=payload, verify=False)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError("❌ API hatası:", res.text)

# =============================== #
# JSON'dan Flat Dict Oluştur
# =============================== #
def parse_and_flatten(json_text: str):
    data = json.loads(json_text)

    flat_doc = {
        "id": data.get("id"),
        "ad_soyad": data.get("ad_soyad"),
        "telefon": data.get("iletişim_bilgileri", {}).get("telefon"),
        "email": data.get("iletişim_bilgileri", {}).get("email"),
        "okul_1": data.get("eğitim_bilgileri", [{}])[0].get("okul_adı"),
        "bolum_1": data.get("eğitim_bilgileri", [{}])[0].get("bölüm"),
        "okul_2": data.get("eğitim_bilgileri", [{}])[1].get("okul_adı") if len(data.get("eğitim_bilgileri", [])) > 1 else None,
        "firma_1": data.get("iş_deneyimleri", [{}])[0].get("firma_adı"),
        "pozisyon_1": data.get("iş_deneyimleri", [{}])[0].get("pozisyon"),
        "teknolojiler": ", ".join(data.get("kullanılan_teknolojiler", []))
    }

    return flat_doc

# =============================== #
# Ana Süreç
# =============================== #
def process_cv(pdf_path, api_key):
    print("🔍 PDF'ten metin çıkarılıyor...")
    metin = extract_text_from_pdf(pdf_path)
    print("✅ Metin bulundu.")

    prompt = f"""
Aşağıdaki CV metninden json formatında yapısal bilgi çıkar:
- ad_soyad
- iletişim_bilgileri: telefon, email
- eğitim_bilgileri: okul_adı, bölüm, başlangıç, bitiş
- iş_deneyimleri: firma_adı, pozisyon, başlangıç, bitiş
- kullanılan_teknolojiler (liste halinde yaz)
Yalnızca geçerli bir JSON döndür:

---
{metin}
---
"""
    print("🤖 LLM'e istek gönderiliyor...")
    yanit = ask_qwen_openrouter(prompt, api_key)
    print("🧠 Model Yanıtı:")
    print(yanit)

    try:
        doc = parse_and_flatten(yanit)
        collection.update_one({"id": doc["id"]}, {"$set": doc}, upsert=True)
        print("✅ MongoDB'ye kayıt başarılı.")
    except Exception as e:
        print("❌ Kayıt sırasında hata:", e)

# =============================== #
# Çalıştır
# =============================== #
if __name__ == "__main__":
    OPENROUTER_API_KEY="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"
    process_cv(PDF_PATH, OPENROUTER_API_KEY)
