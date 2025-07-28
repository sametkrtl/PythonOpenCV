from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os, tempfile, uuid, re, random, fitz
from faker import Faker

app = FastAPI()

# CORS ayarı (frontend'den erişim için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükle
model_path = "./ner_model_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Renkli etiket bilgileri (gerekmese bile dursun)
label_colors = {
    "sirket": (0.56, 0.93, 0.56), "tarih": (0.68, 0.85, 0.90), "ad_soyad": (1.0, 0.71, 0.76),
    "para": (0.94, 0.90, 0.55), "adres": (0.94, 0.50, 0.50), "telefon": (0.87, 0.63, 0.87),
    "tc_kimlik": (1.0, 0.65, 0.0)
}
faker = Faker("tr_TR")
para_birimleri = ["TL", "₺", "Türk Lirası"]
TC_KIMLIK_REGEX = r"\b[1-9]\d{10}\b"

@app.post("/sansurlu_metin")
def sansurlu_metin(metin: str = Form(...), etiketler: str = Form(...), tc_kurali: bool = Form(True)):
    etiketler = [e.strip() for e in etiketler.split(",")]
    entities = ner_pipeline(metin)
    if tc_kurali:
        for match in re.finditer(TC_KIMLIK_REGEX, metin):
            tc = match.group(0)
            if valid_tc(tc):
                entities.append({'entity_group': 'tc_kimlik', 'start': match.start(), 'end': match.end()})

    result = ""
    last = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        label = ent["entity_group"]
        if label not in etiketler:
            continue
        start, end = ent["start"], ent["end"]
        result += metin[last:start] + "*" * (end - start)
        last = end
    result += metin[last:]
    return {"output": result}

@app.post("/sansursuz_metin")
def sansursuz_metin(metin: str = Form(...), etiketler: str = Form(...), tc_kurali: bool = Form(True)):
    etiketler = [e.strip() for e in etiketler.split(",")]
    entities = ner_pipeline(metin)
    if tc_kurali:
        for match in re.finditer(TC_KIMLIK_REGEX, metin):
            tc = match.group(0)
            if valid_tc(tc):
                entities.append({'entity_group': 'tc_kimlik', 'start': match.start(), 'end': match.end()})

    result = ""
    last = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        label = ent["entity_group"]
        if label not in etiketler:
            continue
        start, end = ent["start"], ent["end"]
        fake = fake_value(label, end - start)
        result += metin[last:start] + fake
        last = end
    result += metin[last:]
    return {"output": result}

@app.post("/sansurlu_pdf")
async def sansurlu_pdf(file: UploadFile = File(...), etiketler: str = Form(...), tc_kurali: bool = Form(True)):
    return await process_pdf(file, etiketler, tc_kurali, sahte_kullan=False)

@app.post("/sansursuz_pdf")
async def sansursuz_pdf(file: UploadFile = File(...), etiketler: str = Form(...), tc_kurali: bool = Form(True)):
    return await process_pdf(file, etiketler, tc_kurali, sahte_kullan=True)

# PDF işlemleyici (içeride tutulur)
async def process_pdf(file, etiketler, tc_kurali, sahte_kullan):
    etiketler = [e.strip() for e in etiketler.split(",")]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    doc = fitz.open(temp_path)
    for page in doc:
        words = page.get_text("words")
        if not words:
            continue
        full_text = " ".join(w[4] for w in words)
        entities = ner_pipeline(full_text)
        if tc_kurali:
            for match in re.finditer(TC_KIMLIK_REGEX, full_text):
                tc = match.group(0)
                if valid_tc(tc):
                    entities.append({'entity_group': 'tc_kimlik', 'start': match.start(), 'end': match.end()})
        for ent in entities:
            if ent["entity_group"] not in etiketler:
                continue
            span = full_text[ent["start"]:ent["end"]].lower()
            for w in words:
                kelime = w[4].strip().lower()
                if span in kelime or kelime in span:
                    rect = fitz.Rect(w[0], w[1], w[2], w[3])
                    new_text = fake_value(ent["entity_group"], len(kelime)) if sahte_kullan else "*" * len(kelime)
                    cover_text(page, rect, new_text)
                    break
    output_path = os.path.join(tempfile.gettempdir(), f"pdf_{uuid.uuid4().hex}.pdf")
    doc.save(output_path)
    doc.close()
    return FileResponse(output_path, media_type="application/pdf", filename="output.pdf")

# Yardımcılar (sadece içeride)
def valid_tc(tc):
    if not tc.isdigit() or len(tc) != 11 or int(tc[0]) == 0:
        return False
    d = [int(x) for x in tc]
    return ((sum(d[0:9:2])*7 - sum(d[1:8:2])) % 10 == d[9]) and (sum(d[:10]) % 10 == d[10])

def fake_value(label, uzunluk=None):
    if label == "ad_soyad":
        val = f"{faker.first_name()} {faker.last_name()}"
    elif label == "sirket":
        val = faker.company()
    elif label == "adres":
        val = faker.street_address()
    elif label == "telefon":
        val = f"+90{random.choice(['501','532','555'])}{random.randint(1000000, 9999999)}"
    elif label == "tc_kimlik":
        digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
        odd = sum(digits[0:9:2])
        even = sum(digits[1:8:2])
        d10 = ((odd * 7) - even) % 10
        d11 = (sum(digits[:9]) + d10) % 10
        val = "".join(map(str, digits + [d10, d11]))
    elif label == "para":
        val = f"{random.randint(100,9999)} {random.choice(para_birimleri)}"
    elif label == "tarih":
        tarih = faker.date_between(start_date="-10y", end_date="today")
        val = tarih.strftime("%d.%m.%Y")
    else:
        val = "[MASK]"
    return val[:uzunluk].ljust(uzunluk) if uzunluk else val

def cover_text(page, rect, new_text, font_size=10):
    page.draw_rect(fitz.Rect(rect.x0-1, rect.y0-1, rect.x1+1, rect.y1+1), fill=(1,1,1))
    page.insert_textbox(rect, new_text, fontsize=font_size, fontname="helv", color=(0,0,0))
