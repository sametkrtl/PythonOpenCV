import pandas as pd
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from huggingface_hub import login

# --- Hugging Face Token ve Model Bilgileri ---
hf_token = "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"  # Buraya kendi geçerli HF token'ını yapıştır
model_name = "savasy/bert-base-turkish-ner-cased"

# --- HF Girişi ---
try:
    login(token=hf_token)
except Exception as e:
    print(f"!!! Hugging Face oturum açma başarısız: {e}")
    exit()

# --- Model ve Pipeline Yükleme ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForTokenClassification.from_pretrained(model_name, token=hf_token)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    print("--- Model başarıyla yüklendi ve hazır. ---")
except Exception as e:
    print(f"HATA: Model indirilirken/yüklenirken bir sorun oluştu: {e}")
    exit()

# --- Sayı Sözcükleri ve Regex Pattern'leri ---
number_words = {
    'bir': 1, 'iki': 2, 'üç': 3, 'dört': 4, 'beş': 5, 'altı': 6, 'yedi': 7, 'sekiz': 8, 'dokuz': 9,
    'on': 10, 'yirmi': 20, 'otuz': 30, 'kırk': 40, 'elli': 50, 'altmış': 60, 'yetmiş': 70, 'seksen': 80, 'doksan': 90,
    'yüz': 100, 'bin': 1000, 'milyon': 1000000
}
number_words_pattern = '|'.join(number_words.keys())

REGEX_PATTERNS = {
    "tc_kimlik": re.compile(r'\b[1-9][0-9]{10}\b'),
    "telefon": re.compile(
        r'(?:'
        r'(?:(?:\+?90|0090|0)?[\s\-\.]?)?'           # +90, 0090, 0 (opsiyonel)
        r'(?:\(?\d{3,4}\)?)[\s\-\.]?'                # alan kodu
        r'\d{3}[\s\-\.]?\d{2}[\s\-\.]?\d{2}'          # numara
        r')'
        r'|'
        r'(?:'
        r'(?:\+?90|0090)?0?\d{10}'                   # +905451234567, 05451234567, 905451234567
        r')'
        r'|'
        r'(?:'
        r'(?:444|850)[\s\-\.]?\d{3}[\s\-\.]?\d{4}'    # 444 xxx xxx veya 850 xxx xxxx
        r')',
        re.IGNORECASE
    ),
    "para": re.compile(
        r'((?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:,\d{1,2})?\s*(?:TL|₺|Dolar|Euro|€|\$|lira|liralık)\b)|'
        r'(\b(?:' + number_words_pattern + r')(?:\s+(?:' + number_words_pattern + r'))*\s+(?:Türk\sLirası|lira|TL|₺)\b)',
        re.IGNORECASE
    ),
    "tarih": re.compile(
        r'(\b\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}\b)|'
        r'(\b\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{2,4}\b)|'
        r'(\b(?:geçen\s+hafta|bugün|yarın|dün)\b)',
        re.IGNORECASE
    )
}

LABEL_MAP = {
    "PER": "ad_soyad",
    "ORG": "sirket",
    "LOC": "adres"
}

# --- TC Kimlik kontrolü ---
def is_valid_tc(tc_string: str) -> bool:
    tc_string = re.sub(r'\s', '', tc_string)
    if not tc_string.isdigit() or len(tc_string) != 11 or int(tc_string[0]) == 0:
        return False
    digits = [int(d) for d in tc_string]
    if ((sum(digits[0:9:2]) * 7 - sum(digits[1:8:2])) % 10) != digits[9]:
        return False
    if (sum(digits[0:10]) % 10) != digits[10]:
        return False
    return True

# --- Regex + Model ile Etiketleme ---
def find_entities(text):
    all_entities = []

    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            matched_text = match.group(0)

            # TC kimlik kontrolü
            if label == "tc_kimlik" and not is_valid_tc(matched_text):
                continue

            # Telefon numarası için 11 rakam kontrolü
            if label == "telefon":
                digit_count = len(re.sub(r"\D", "", matched_text))
                if digit_count < 11:
                    continue

            all_entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": label,
                "source": "regex"
            })

    try:
        ner_results = ner_pipeline(text)
        for ent in ner_results:
            if ent["entity_group"] in LABEL_MAP:
                all_entities.append({
                    "start": ent["start"],
                    "end": ent["end"],
                    "label": LABEL_MAP[ent["entity_group"]],
                    "source": "model"
                })
    except Exception:
        pass

    # Örtüşme temizliği
    if not all_entities:
        return []

    all_entities.sort(key=lambda x: x['start'])
    final = []
    prev = all_entities[0]

    for current in all_entities[1:]:
        if current["start"] < prev["end"]:
            if prev["source"] == "regex" and current["source"] == "model":
                continue
            if current["end"] > prev["end"]:
                prev = current
            continue
        final.append(prev)
        prev = current

    final.append(prev)
    return [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in final]


# --- Ana Dönüştürme Fonksiyonu ---
def process_excel_json_lines(input_excel_path, output_jsonl_path):
    try:
        df = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"HATA: Excel dosyası okunurken sorun oluştu: {e}")
        return

    if "metin" not in df.columns:
        print("HATA: Excel dosyasında 'metin' adlı bir sütun yok.")
        return

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for i, metin in tqdm(df["metin"].items(), total=df.shape[0], desc="Satırlar işleniyor"):
            if not isinstance(metin, str) or not metin.strip():
                continue
            try:
                entities = find_entities(metin)
                json_line = {"text": metin.strip(), "entities": entities}
                outfile.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[{i}] Satır işlenemedi: {e}")
                continue

    print(f"\n✅ Tamamlandı: JSONL çıktısı → '{output_jsonl_path}'")

# --- Çalıştırma Bölümü ---
if __name__ == "__main__":
    INPUT_XLSX = "/content/data.xlsx"         # Giriş Excel dosyası yolu
    OUTPUT_JSONL = "etiketli_veri.jsonl"           # Çıkış dosyası adı
    process_excel_json_lines(INPUT_XLSX, OUTPUT_JSONL)
