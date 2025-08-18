import gradio as gr
import os
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import logging
from faker import Faker
import random
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import fitz  # PyMuPDF
import re
import numpy as np  # <-- EKLE
# ----------------------------------------------------
#  EnhancedAnonymizationApp  (Custom NER Model + Font Preservation)
# ----------------------------------------------------
class EnhancedAnonymizationApp:
    def __init__(self):
        # Logger ayarları
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model yolunu belirtin
        self.model_path = r"C:\Users\stj.skartal\Desktop\python\samet\ner_model_final"
        
        # NER pipeline'ını yükle
        self.ner_pipeline = self.load_custom_ner_model()
        
        # Tutarlı sahte üretim için sabit seed
        self.global_seed = 123456
        random.seed(self.global_seed)

        # 🇹🇷 Türkçe Faker örneği
        self.fake = Faker("tr_TR")
        self.fake.seed_instance(self.global_seed)

        # Dizinleri oluştur
        for dir_name in ["uploads", "outputs", "logs", "temp"]:
            os.makedirs(dir_name, exist_ok=True)
        
        self.processing_stats = []
    
    def load_custom_ner_model(self):
        """Özel NER modelini yükle"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model bulunamadı: {self.model_path}")
                return None
            
            self.logger.info(f"Model yükleniyor: {self.model_path}")
            
            # Tokenizer ve model yükle
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            # Pipeline oluştur
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("Model başarıyla yüklendi!")
            return ner_pipeline
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return None
    
    def extract_text_with_positions(self, pdf_path: str) -> List[Dict]:
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text = (span.get("text") or "").strip()
                                if text:
                                    # --- TİP DÖNÜŞÜMLERİ: float32 → float, vs. ---
                                    bbox_raw = span.get("bbox", (0, 0, 0, 0))
                                    # fitz Rect ya da tuple olabilir
                                    if hasattr(bbox_raw, "__iter__"):
                                        bbox = tuple(float(x) for x in bbox_raw)
                                    else:
                                        # Rect ise
                                        try:
                                            bbox = (float(bbox_raw.x0), float(bbox_raw.y0),
                                                    float(bbox_raw.x1), float(bbox_raw.y1))
                                        except Exception:
                                            bbox = (0.0, 0.0, 0.0, 0.0)

                                    size = float(span.get("size", 12))
                                    flags = int(span.get("flags", 0))
                                    color = span.get("color", 0)
                                    # color bazen float olabilir; int'e güvenli döndür
                                    try:
                                        color = int(color)
                                    except Exception:
                                        color = 0

                                    text_blocks.append({
                                        'page': int(page_num),
                                        'text': text,
                                        'bbox': bbox,               # artık saf Python float tuple
                                        'font': str(span.get("font") or "Unknown"),
                                        'size': size,
                                        'flags': flags,
                                        'color': color,
                                        'start_char': 0,
                                        'end_char': 0
                                    })
            doc.close()

            # global pozisyonlar
            char_position = 0
            for block in text_blocks:
                block['start_char'] = int(char_position)
                char_position += len(block['text']) + 1
                block['end_char'] = int(char_position - 1)

            return text_blocks
        except Exception as e:
            self.logger.error(f"Metin çıkarma hatası: {e}")
            return []

     # --- EKLE: JSON güvenli dönüştürücü ---
    def _json_safe(self, obj):
        # numpy sayıları
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # torch tensor
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        # pandas zaman tipleri
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # diğer yaygın yapılar
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", "ignore")
        # fitz.Rect veya benzeri
        try:
            from fitz import Rect
            if isinstance(obj, Rect):
                return [float(obj.x0), float(obj.y0), float(obj.x1), float(obj.y1)]
        except Exception:
            pass
        # son çare: str
        return str(obj)
    def process_pdf_with_real_replacement(self, 
                                        pdf_file, 
                                        confidence_threshold: float,
                                        replacement_strategy: str,
                                        custom_replacements: str,
                                        enable_logging: bool,
                                        enable_debug: bool,
                                        conversion_method: str,
                                        progress=gr.Progress()) -> Tuple[Optional[str], str, str, pd.DataFrame, str]:
        """
        Gelişmiş PDF işleme - GERÇEK değiştirme ile (Custom NER + Font Preservation)
        """
        if pdf_file is None:
            return None, "❌ Lütfen bir PDF dosyası yükleyin.", "", pd.DataFrame(), ""
        
        if self.ner_pipeline is None:
            return None, "❌ NER modeli yüklenemedi. Model yolunu kontrol edin.", "", pd.DataFrame(), ""
        
        debug_info = ""
        
        try:
            progress(0.05, desc="Dosya hazırlanıyor...")
            
            # Dosya yolları
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_in = os.path.basename(pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file))
            input_filename = f"input_{timestamp}_{base_in}"
            output_filename = f"anonymized_{timestamp}_{base_in}"
            
            input_path = os.path.join("uploads", input_filename)
            output_path = os.path.join("outputs", output_filename)
            
            # Yüklenen dosyayı kopyala
            shutil.copy2(pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file), input_path)
            
            progress(0.1, desc="PDF analiz ediliyor...")
            
            # 1. PDF'ten metin ve pozisyonları çıkar
            text_blocks = self.extract_text_with_positions(input_path)
            full_text = " ".join([block['text'] for block in text_blocks])
            
            if not full_text.strip():
                return None, "⚠️ PDF'den metin çıkarılamadı.", "", pd.DataFrame(), "Metin çıkarılamadı."
            
            progress(0.2, desc="NER analizi yapılıyor...")
            
            # 2. Custom NER model ile analiz
            entities_detected = self.extract_entities_with_custom_model(full_text, text_blocks, confidence_threshold, progress)
            
            if not entities_detected:
                return None, "⚠️ PDF'de kişisel bilgi tespit edilmedi.", "", pd.DataFrame(), "Hiçbir entity tespit edilmedi."
            
            progress(0.4, desc="Değiştirme stratejisi uygulanıyor...")
            
            # 3. Değiştirme stratejisini uygula
            processed_entities = self.apply_replacement_strategy(
                entities_detected, 
                replacement_strategy, 
                custom_replacements or ""
            )
            
            progress(0.5, desc="PDF'te değişiklikler uygulanıyor...")
            
            # 4. Font korumalı PDF değiştirme
            success = self.perform_font_preserving_replacement(
                input_path, 
                processed_entities, 
                output_path, 
                text_blocks,
                progress
            )
            
            if not success:
                return None, "❌ PDF değiştirme işlemi başarısız oldu.", "", pd.DataFrame(), "Değiştirme başarısız"
            
            progress(0.8, desc="Sonuçlar doğrulanıyor...")
            
            # 5. Değiştirmeleri doğrula
            validation_result = self.validate_pdf_changes(output_path, processed_entities)
            debug_info = self.format_debug_info(validation_result, processed_entities)
            
            progress(0.9, desc="İstatistikler oluşturuluyor...")
            
            # 6. İstatistikleri oluştur
            stats_df = self.create_detailed_statistics(processed_entities, validation_result)
            summary = self.generate_detailed_summary(processed_entities, validation_result, input_filename)
            
            # 7. Loglama
            if enable_logging:
                self.log_processing_with_validation(input_filename, processed_entities, validation_result)
            
            progress(1.0, desc="Tamamlandı!")
            
            # Başarı mesajı
            success_rate = validation_result.get('successfully_replaced', 0)
            total_entities = len(processed_entities)
            status_msg = f"✅ İşlem tamamlandı! {success_rate}/{total_entities} değiştirme başarılı."
            
            return output_path, status_msg, summary, stats_df, (debug_info if enable_debug else "")
            
        except Exception as e:
            self.logger.error(f"PDF işleme hatası: {e}", exc_info=True)
            error_msg = f"❌ Kritik hata: {str(e)}"
            return None, error_msg, "", pd.DataFrame(), f"Hata detayı: {str(e)}"
    
    def extract_entities_with_custom_model(self, full_text: str, text_blocks: List[Dict], 
                                         confidence_threshold: float, progress) -> List[Dict]:
        """Custom NER model ile entity tespiti"""
        try:
            # Metni parçalara böl (model token limitini aşmamak için)
            max_length = 512
            text_chunks = []
            chunk_start = 0
            
            while chunk_start < len(full_text):
                chunk_end = min(chunk_start + max_length, len(full_text))
                
                # Kelime ortasında kesmemek için
                if chunk_end < len(full_text):
                    last_space = full_text.rfind(' ', chunk_start, chunk_end)
                    if last_space > chunk_start:
                        chunk_end = last_space
                
                chunk_text = full_text[chunk_start:chunk_end]
                text_chunks.append({
                    'text': chunk_text,
                    'start_offset': chunk_start,
                    'end_offset': chunk_end
                })
                chunk_start = chunk_end
            
            progress(0.25, desc=f"NER analizi ({len(text_chunks)} parça)...")
            
            all_entities = []
            
            # Her parça için NER çalıştır
            for i, chunk in enumerate(text_chunks):
                try:
                    results = self.ner_pipeline(chunk['text'])
                    
                    for result in results:
                        if result['score'] >= confidence_threshold:
                            # Global pozisyonu hesapla
                            entity_start = chunk['start_offset'] + result['start']
                            entity_end = chunk['start_offset'] + result['end']
                            
                            # Text block'ta karşılık gelen pozisyonu bul
                            text_block_info = self.find_text_block_for_position(
                                entity_start, entity_end, text_blocks, full_text
                            )
                            
                            entity_dict = {
                                'entity': self.map_model_label_to_type(result['entity_group']),
                                'word': result['word'],
                                'start': entity_start,
                                'end': entity_end,
                                'score': result['score'],
                                'method': 'custom_ner',
                                'text_block_info': text_block_info
                            }
                            all_entities.append(entity_dict)
                
                except Exception as e:
                    self.logger.warning(f"Chunk {i} işlenirken hata: {e}")
                    continue
            
            progress(0.3, desc="Regex ile ek kontroller...")
            
            # Regex ile ek tespitler (TC kimlik vb.)
            regex_entities = self.perform_regex_detection_with_blocks(full_text, text_blocks)
            
            # Birleştir ve temizle
            combined_entities = self.merge_and_clean_entities(all_entities + regex_entities, confidence_threshold)
            
            self.logger.info(f"Toplam {len(combined_entities)} entity tespit edildi")
            return combined_entities
            
        except Exception as e:
            self.logger.error(f"Custom NER analiz hatası: {e}")
            return []
    
    def map_model_label_to_type(self, model_label: str) -> str:
        """Model etiketlerini uygulama türlerine eşle"""
        # Modelinizin çıktı etiketlerine göre bu mapping'i güncelleyin
        label_mapping = {
            'PERSON': 'ad_soyad',
            'PER': 'ad_soyad',
            'B-PERSON': 'ad_soyad',
            'I-PERSON': 'ad_soyad',
            'PHONE': 'telefon',
            'PHONE_NUMBER': 'telefon',
            'EMAIL': 'email',
            'ADDRESS': 'adres',
            'ORGANIZATION': 'sirket',
            'ORG': 'sirket',
            'MONEY': 'para',
            'DATE': 'tarih',
            'ID_NUMBER': 'tc_kimlik',
            'NATIONAL_ID': 'tc_kimlik'
        }
        return label_mapping.get(model_label.upper(), model_label.lower())
    
    def find_text_block_for_position(self, start_pos: int, end_pos: int, 
                                   text_blocks: List[Dict], full_text: str) -> Dict:
        """Verilen pozisyon için text block bilgilerini bul"""
        current_pos = 0
        
        for block in text_blocks:
            block_text = block['text']
            block_start = current_pos
            block_end = current_pos + len(block_text)
            
            # Entity bu block içinde mi?
            if start_pos >= block_start and end_pos <= block_end + 1:
                # Block içindeki relatif pozisyonu hesapla
                relative_start = start_pos - block_start
                relative_end = end_pos - block_start
                
                return {
                    'page': block['page'],
                    'bbox': block['bbox'],
                    'font': block['font'],
                    'size': block['size'],
                    'flags': block['flags'],
                    'color': block['color'],
                    'relative_start': relative_start,
                    'relative_end': relative_end,
                    'block_text': block_text
                }
            
            current_pos = block_end + 1  # +1 for space/newline
        
        # Bulunamadı ise default değer
        return {
            'page': 0,
            'bbox': (0, 0, 100, 20),
            'font': 'Arial',
            'size': 12,
            'flags': 0,
            'color': 0,
            'relative_start': 0,
            'relative_end': 0,
            'block_text': ''
        }
    
    def perform_regex_detection_with_blocks(self, full_text: str, text_blocks: List[Dict]) -> List[Dict]:
        """Regex ile ek tespitler (pozisyon bilgisiyle)"""
        regex_entities = []
        
        # TC Kimlik pattern
        tc_pattern = r'\b[1-9][0-9]{9}[02468]\b'
        for match in re.finditer(tc_pattern, full_text):
            tc_no = match.group()
            if self.validate_turkish_id(tc_no):
                text_block_info = self.find_text_block_for_position(
                    match.start(), match.end(), text_blocks, full_text
                )
                
                regex_entities.append({
                    'entity': 'tc_kimlik',
                    'word': tc_no,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.95,
                    'method': 'regex_validated',
                    'text_block_info': text_block_info
                })
        
        # Telefon pattern
        phone_pattern = r'\b0?5[0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}\b'
        for match in re.finditer(phone_pattern, full_text):
            text_block_info = self.find_text_block_for_position(
                match.start(), match.end(), text_blocks, full_text
            )
            
            regex_entities.append({
                'entity': 'telefon',
                'word': match.group(),
                'start': match.start(),
                'end': match.end(),
                'score': 0.85,
                'method': 'regex',
                'text_block_info': text_block_info
            })
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, full_text):
            text_block_info = self.find_text_block_for_position(
                match.start(), match.end(), text_blocks, full_text
            )
            
            regex_entities.append({
                'entity': 'email',
                'word': match.group(),
                'start': match.start(),
                'end': match.end(),
                'score': 0.90,
                'method': 'regex',
                'text_block_info': text_block_info
            })
        
        return regex_entities
    
    def perform_font_preserving_replacement(self, input_path: str, entities: List[Dict], 
                                          output_path: str, text_blocks: List[Dict], progress) -> bool:
        """Font özelliklerini koruyarak PDF değiştirme"""
        try:
            doc = fitz.open(input_path)
            total_replacements = 0
            
            progress(0.6, desc="Font bilgileri korunarak değiştiriliyor...")
            
            # Sayfa bazında entity'leri grupla
            entities_by_page = {}
            for entity in entities:
                page_num = entity.get('text_block_info', {}).get('page', 0)
                if page_num not in entities_by_page:
                    entities_by_page[page_num] = []
                entities_by_page[page_num].append(entity)
            
            # Her sayfa için işlem yap
            for page_num in range(len(doc)):
                if page_num not in entities_by_page:
                    continue
                
                page = doc.load_page(page_num)
                page_entities = entities_by_page[page_num]
                
                progress(0.6 + (page_num / len(doc)) * 0.2, 
                        desc=f"Sayfa {page_num + 1}/{len(doc)} işleniyor...")
                
                # Entity'leri pozisyona göre sırala (tersten - sondan başa)
                page_entities.sort(key=lambda x: x.get('start', 0), reverse=True)
                
                for entity in page_entities:
                    success = self.replace_entity_with_font_preservation(page, entity)
                    if success:
                        total_replacements += 1
            
            doc.save(output_path)
            doc.close()
            
            self.logger.info(f"Font korumalı değiştirme: {total_replacements} başarılı")
            return total_replacements > 0
            
        except Exception as e:
            self.logger.error(f"Font korumalı değiştirme hatası: {e}")
            return False
    
    def replace_entity_with_font_preservation(self, page, entity: Dict) -> bool:
        """Tek bir entity'yi font özelliklerini koruyarak değiştir"""
        try:
            original_text = entity.get('word', '').strip()
            replacement_text = entity.get('replacement', original_text).strip()
            
            if not original_text or original_text == replacement_text:
                return False
            
            text_block_info = entity.get('text_block_info', {})
            bbox = text_block_info.get('bbox')
            
            if not bbox:
                self.logger.warning(f"Entity için bbox bulunamadı: {original_text}")
                return False
            
            # Orijinal metni bul ve pozisyonunu al
            text_instances = page.search_for(original_text)
            
            if not text_instances:
                self.logger.warning(f"Metin sayfada bulunamadı: {original_text}")
                return False
            
            # En yakın bbox'ı bul
            target_rect = None
            min_distance = float('inf')
            
            for rect in text_instances:
                # Bbox'lar arasındaki mesafeyi hesapla
                distance = abs(rect[0] - bbox[0]) + abs(rect[1] - bbox[1])
                if distance < min_distance:
                    min_distance = distance
                    target_rect = rect
            
            if target_rect is None:
                return False
            
            # Font özelliklerini al (sadece boyut ve renk)
            font_size = text_block_info.get('size', 12)
            font_color = text_block_info.get('color', 0)
            
            # Orijinal metni sil (beyaz dikdörtgen ile kapat)
            page.add_redact_annot(target_rect, fill=(1, 1, 1))
            page.apply_redactions()
            
            # Yeni metin için pozisyon ayarla
            rect_width = target_rect[2] - target_rect[0]
            rect_height = target_rect[3] - target_rect[1]
            
            # Default font ile metin boyutunu hesapla
            default_font = 'helv'  # Helvetica (PDF standard font)
            text_width = fitz.get_text_length(replacement_text, fontname=default_font, fontsize=font_size)
            
            # Metin çok uzunsa font boyutunu küçült
            if text_width > rect_width * 1.1:  # %10 tolerans
                adjusted_font_size = font_size * (rect_width / text_width) * 0.9
                font_size = max(adjusted_font_size, 6)  # Minimum 6pt
            
            # RGB renk dönüştürme
            if isinstance(font_color, int):
                # Integer color'ı RGB'ye çevir
                rgb_color = (
                    ((font_color >> 16) & 255) / 255.0,
                    ((font_color >> 8) & 255) / 255.0,
                    (font_color & 255) / 255.0
                )
            else:
                rgb_color = (0, 0, 0)  # Siyah default
            
            # Metni ekle (default font ile)
            insert_point = (target_rect[0], target_rect[3] - 2)  # Biraz yukarıdan
            
            page.insert_text(
                insert_point,
                replacement_text,
                fontsize=font_size,
                fontname=default_font,  # Default font kullan
                color=rgb_color,
                render_mode=0  # Normal text
            )
            
            self.logger.debug(f"Default font ile değiştirme: '{original_text}' -> '{replacement_text}' (Font: Helvetica, Boyut: {font_size})")
            return True
            
        except Exception as e:
            self.logger.error(f"Entity değiştirme hatası: {e}")
            return False
    
    def validate_turkish_id(self, tc_no: str) -> bool:
        """TC kimlik validasyonu"""
        if not tc_no or len(tc_no) != 11 or not tc_no.isdigit():
            return False
        if tc_no[0] == '0':
            return False
        digits = [int(d) for d in tc_no]
        odd_sum = sum(digits[i] for i in range(0, 9, 2))
        even_sum = sum(digits[i] for i in range(1, 8, 2))
        check_digit_10 = (odd_sum * 7 - even_sum) % 10
        if check_digit_10 != digits[9]:
            return False
        check_digit_11 = sum(digits[:10]) % 10
        if check_digit_11 != digits[10]:
            return False
        return True
    
    def merge_and_clean_entities(self, entities: List[Dict], confidence_threshold: float) -> List[Dict]:
        """Entity'leri birleştir ve temizle"""
        seen = set()
        cleaned = []
        for entity in entities:
            key = (entity.get('start', 0), entity.get('end', 0), entity.get('word', ''))
            if key not in seen:
                seen.add(key)
                cleaned.append(entity)
        filtered = [e for e in cleaned if e.get('score', 0) >= confidence_threshold]
        return sorted(filtered, key=lambda x: x.get('start', 0))
    
    # ------------ Faker yardımcıları ------------
    def _seeded_faker(self, key: str) -> Faker:
        """Aynı 'key' için deterministik Faker."""
        seeded = abs(hash(key)) % (10**9)
        f = Faker("tr_TR")
        f.seed_instance(seeded)
        return f

    def _fake_tc_kimlik(self, f: Faker) -> str:
        """Geçerli checksum'lı TR TCKN üret."""
        while True:
            d = [0]*11
            d[0] = f.random_int(1, 9)
            for i in range(1, 9):
                d[i] = f.random_int(0, 9)
            odd_sum = sum(d[i] for i in range(0, 9, 2))
            even_sum = sum(d[i] for i in range(1, 8, 2))
            d[9] = (odd_sum * 7 - even_sum) % 10
            d[10] = (sum(d[:10]) % 10)
            tc = "".join(map(str, d))
            if self.validate_turkish_id(tc):
                return tc

    def _fake_phone_tr(self, f: Faker) -> str:
        """05xxxxxxxxx formatında cep."""
        return "05" + "".join(str(f.random_int(0, 9)) for _ in range(9))

    def _fake_iban_tr(self, f: Faker) -> str:
        try:
            return f.iban(country_code="TR")
        except Exception:
            return "TR" + "".join(str(f.random_int(0, 9)) for _ in range(24))

    def _fake_company_tr(self, f: Faker) -> str:
        try:
            return f.company()
        except Exception:
            return "Örnek A.Ş."

    def _fake_address_tr(self, f: Faker) -> str:
        try:
            return f.address().replace("\n", ", ")
        except Exception:
            return "Örnek Mah., Örnek Cd., No:1, İstanbul"

    def _fake_name_tr(self, f: Faker) -> str:
        try:
            return f.name()
        except Exception:
            return "Ali Demir"

    def _fake_email_from_name(self, f: Faker, name_full: str) -> str:
        import unicodedata, re
        def slugify(s: str) -> str:
            s = unicodedata.normalize("NFKD", s)
            s = "".join(c for c in s if not unicodedata.combining(c))
            s = s.lower()
            s = (s.replace("ı","i").replace("ğ","g").replace("ş","s")
                   .replace("ç","c").replace("ö","o").replace("ü","u"))
            s = re.sub(r"[^a-z0-9]+", ".", s).strip(".")
            return s

        parts = (name_full or "").split()
        if len(parts) >= 2:
            user = f"{slugify(parts[0])}.{slugify(parts[-1])}"
        else:
            user = slugify(name_full or "kullanici")
        domain = f.random_element(["example.com","mail.com","ornek.com","kurum.com.tr"])
        return f"{user}@{domain}"
    
    def apply_replacement_strategy(self, entities: List[Dict], strategy: str, custom_replacements: str) -> List[Dict]:
        """Değiştirme stratejisini uygula (Faker destekli)."""
        default_replacements = {
            'ad_soyad': 'Ali Demir',
            'tc_kimlik': '11111111110',
            'telefon': '05009999999',
            'adres': 'Örnek Mahallesi, İstanbul',
            'para': '1000 TL',
            'tarih': '01.01.2000',
            'email': 'ornek@email.com',
            'sirket': 'Örnek A.Ş.',
            'iban': 'TR00 0000 0000 0000 0000 0000 00'
        }
        if custom_replacements and custom_replacements.strip():
            try:
                custom_dict = json.loads(custom_replacements)
                default_replacements.update(custom_dict)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Custom replacements parse hatası: {e}")

        processed = []
        for entity in entities:
            e = entity.copy()
            etype = e.get('entity', '').strip()
            original = e.get('word', '')

            if strategy == "Maskeleme (*)":
                e['replacement'] = '*' * len(original)

            elif strategy == "Genel Değerler":
                generic_map = {
                    'ad_soyad': '[İSİM]',
                    'tc_kimlik': '[TC KİMLİK]',
                    'telefon': '[TELEFON]',
                    'adres': '[ADRES]',
                    'para': '[PARA]',
                    'tarih': '[TARİH]',
                    'email': '[E-POSTA]',
                    'sirket': '[ŞİRKET]',
                    'iban': '[IBAN]'
                }
                e['replacement'] = generic_map.get(etype, f'[{etype.upper()}]')

            elif strategy == "Sahte Değerler":
                # Aynı orijinal -> aynı sahte değer (deterministik)
                f = self._seeded_faker(original if original else etype)

                # Kullanıcının JSON ile verdiği tür varsa öncelik
                override = default_replacements.get(etype)
                candidate = None if override in (None, "") else override

                if candidate is None:
                    if etype == 'ad_soyad':
                        candidate = self._fake_name_tr(f)
                    elif etype == 'telefon':
                        candidate = self._fake_phone_tr(f)
                    elif etype == 'email':
                        base_name = original if original else self._fake_name_tr(f)
                        candidate = self._fake_email_from_name(f, base_name)
                    elif etype == 'adres':
                        candidate = self._fake_address_tr(f)
                    elif etype == 'sirket':
                        candidate = self._fake_company_tr(f)
                    elif etype == 'iban':
                        candidate = self._fake_iban_tr(f)
                    elif etype == 'tc_kimlik':
                        candidate = self._fake_tc_kimlik(f)
                    elif etype == 'para':
                        tl = f.pydecimal(left_digits=4, right_digits=2, positive=True)
                        candidate = f"{tl} TL"
                    elif etype == 'tarih':
                        dt = f.date_between(start_date="-10y", end_date="today")
                        candidate = dt.strftime("%d.%m.%Y")
                    else:
                        # Tanımsız tür: uzunluğu kabaca koruyan fallback
                        candidate = original if original else "—"

                e['replacement'] = candidate

            else:
                e['replacement'] = original

            processed.append(e)
        return processed
    
    def validate_pdf_changes(self, output_path: str, entities: List[Dict]) -> Dict:
        """PDF'teki değişiklikleri doğrula"""
        try:
            doc = fitz.open(output_path)
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            doc.close()
            
            validation_result = {
                'total_entities': len(entities),
                'successfully_replaced': 0,
                'still_present': [],
                'replacement_found': [],
                'partial_replacements': []
            }
            for entity in entities:
                original = entity.get('word', '').strip()
                replacement = entity.get('replacement', original).strip()
                if original in full_text:
                    validation_result['still_present'].append(original)
                else:
                    validation_result['successfully_replaced'] += 1
                if replacement in full_text and replacement != original:
                    validation_result['replacement_found'].append(replacement)
            return validation_result
        except Exception as e:
            self.logger.error(f"Validasyon hatası: {e}")
            return {'error': str(e)}
    
    def format_debug_info(self, validation_result: Dict, entities: List[Dict]) -> str:
        """Debug bilgilerini formatla"""
        if 'error' in validation_result:
            return f"❌ Validasyon hatası: {validation_result['error']}"
        
        debug_info = f"""
## 🔍 Debug Bilgileri

### Değiştirme Sonuçları:
- **Toplam Entity**: {validation_result['total_entities']}
- **Başarılı Değiştirme**: {validation_result['successfully_replaced']}
- **Hala Mevcut**: {len(validation_result.get('still_present', []))}
- **Yeni Değerler Bulundu**: {len(validation_result.get('replacement_found', []))}

### Hala Mevcut Metinler:
{validation_result.get('still_present', [])[:5]}

### Bulunan Yeni Değerler:
{validation_result.get('replacement_found', [])[:5]}

### Model Bilgileri:
- **Model Yolu**: {self.model_path}
- **Model Durumu**: {'✅ Yüklendi' if self.ner_pipeline else '❌ Yüklenemedi'}
        """.strip()
        return debug_info
    
    def create_detailed_statistics(self, entities: List[Dict], validation_result: Dict) -> pd.DataFrame:
        """Detaylı istatistik tablosu"""
        if not entities:
            return pd.DataFrame(columns=['Veri Türü', 'Tespit', 'Değiştirildi', 'Başarı Oranı', 'Örnek'])
        
        stats = {}
        for entity in entities:
            etype = entity['entity']
            if etype not in stats:
                stats[etype] = {'detected': 0, 'examples': [], 'originals': [], 'replacements': []}
            stats[etype]['detected'] += 1
            stats[etype]['originals'].append(entity.get('word', ''))
            stats[etype]['replacements'].append(entity.get('replacement', ''))
            if len(stats[etype]['examples']) < 2:
                stats[etype]['examples'].append(entity.get('word', ''))
        
        rows = []
        names = {
            'ad_soyad': '👤 Ad Soyad',
            'tc_kimlik': '🆔 TC Kimlik', 
            'telefon': '📱 Telefon',
            'adres': '📍 Adres',
            'para': '💰 Para',
            'tarih': '📅 Tarih',
            'email': '📧 E-posta',
            'sirket': '🏢 Şirket'
        }
        for etype, data in stats.items():
            successful = 0
            for original in data['originals']:
                if original not in validation_result.get('still_present', []):
                    successful += 1
            success_rate = (successful / data['detected'] * 100) if data['detected'] > 0 else 0
            rows.append({
                'Veri Türü': names.get(etype, etype.title()),
                'Tespit': data['detected'],
                'Değiştirildi': successful,
                'Başarı Oranı': f"{success_rate:.1f}%",
                'Örnek': ', '.join(data['examples'][:2])
            })
        return pd.DataFrame(rows)
    
    def generate_detailed_summary(self, entities: List[Dict], validation_result: Dict, filename: str) -> str:
        """Detaylı özet oluştur"""
        if not entities:
            return "❌ Hiçbir kişisel bilgi tespit edilmedi."
        
        total_entities = len(entities)
        successful = validation_result.get('successfully_replaced', 0)
        still_present = len(validation_result.get('still_present', []))
        success_rate = (successful / total_entities * 100) if total_entities > 0 else 0
        
        if success_rate >= 90:
            status_emoji = "🟢"; status_text = "Mükemmel"
        elif success_rate >= 70:
            status_emoji = "🟡"; status_text = "İyi"
        else:
            status_emoji = "🔴"; status_text = "Dikkat Gerekli"
        
        summary = f"""
## 📊 Detaylı İşlem Raporu

**Dosya:** {filename}  
**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}

### {status_emoji} Genel Başarı: {status_text} ({success_rate:.1f}%)

### 🎯 Değiştirme Sonuçları
- **Toplam Tespit:** {total_entities} adet
- **Başarılı Değiştirme:** {successful} adet  
- **Hala Mevcut:** {still_present} adet

### 📈 Performans Analizi
""".strip()
        if success_rate >= 90:
            summary += "\n\n✅ **Harika!** Neredeyse tüm kişisel bilgiler başarıyla değiştirildi."
        elif success_rate >= 70:
            summary += "\n\n⚠️ **İyi performans** ancak bazı bilgiler değiştirilemedi. Manuel kontrol önerilir."
        else:
            summary += "\n\n🚨 **Dikkat!** Çoğu bilgi değiştirilemedi. Farklı yöntem denemeyi düşünün."
        
        if still_present > 0:
            still_present_list = validation_result.get('still_present', [])[:3]
            summary += "\n\n### 🔍 Değiştirilemeyen Örnekler:\n"
            for item in still_present_list:
                summary += f"- `{item}`\n"
        return summary
    
    def log_processing_with_validation(self, filename: str, entities: List[Dict], validation_result: Dict):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'total_entities': len(entities),
            'successful_replacements': validation_result.get('successfully_replaced', 0),
            'still_present_count': len(validation_result.get('still_present', [])),
            'success_rate': (validation_result.get('successfully_replaced', 0) / len(entities) * 100) if entities else 0,
            'entities': entities,
            'validation_result': validation_result
        }
        log_file = f"logs/detailed_session_{datetime.now().strftime('%Y%m%d')}.json"
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            # --- BURADA default=self._json_safe EKLENDİ ---
            json.dump(logs, f, ensure_ascii=False, indent=2, default=self._json_safe)

    
    def get_processing_history(self) -> pd.DataFrame:
        """İşlem geçmişini al"""
        try:
            log_files = []
            for i in range(7):
                date_str = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
                log_file = f"logs/detailed_session_{date_str}.json"
                if os.path.exists(log_file):
                    log_files.append(log_file)
            all_logs = []
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    all_logs.extend(logs)
            if not all_logs:
                return pd.DataFrame(columns=['Tarih', 'Dosya', 'Toplam Tespit', 'Başarılı', 'Başarı Oranı'])
            rows = []
            for log in all_logs[-50:]:
                timestamp = datetime.fromisoformat(log['timestamp'])
                rows.append({
                    'Tarih': timestamp.strftime('%d.%m.%Y %H:%M'),
                    'Dosya': log['filename'],
                    'Toplam Tespit': log['total_entities'],
                    'Başarılı': log.get('successful_replacements', 0),
                    'Başarı Oranı': f"{log.get('success_rate', 0):.1f}%"
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame(columns=['Tarih', 'Dosya', 'Toplam Tespit', 'Başarılı', 'Başarı Oranı'])
    
    # -----------------------------
    # Gradio UI
    # -----------------------------
    def create_enhanced_interface(self):
        """Gelişmiş Gradio arayüzü"""
        css = """
        .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .debug-info { background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; }
        .model-status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .model-loaded { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .model-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        """
        
        with gr.Blocks(css=css, title="PDF Kişisel Bilgi Anonimleştirici - Custom NER", theme=gr.themes.Soft()) as interface:
            # Model durumu göster
            model_status = "✅ Custom NER Model Yüklendi" if self.ner_pipeline else "❌ Model Yüklenemedi"
            model_class = "model-loaded" if self.ner_pipeline else "model-error"
            
            gr.Markdown(f"""
            # 🔐 PDF Kişisel Bilgi Anonimleştirici — Custom NER
            Yüklediğiniz PDF'teki kişisel bilgileri **özel modelinizle tespit eder** ve **font özelliklerini koruyarak değiştirir**.
            
            <div class="model-status {model_class}">
            🤖 **Model Durumu**: {model_status}<br>
            📁 **Model Yolu**: {self.model_path}
            </div>
            """)
            
            with gr.Tabs():
                # ------------------ Ana İşlem Sekmesi ------------------
                with gr.TabItem("📄 PDF İşleme", elem_id="main-tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_input = gr.File(
                                label="📎 PDF Dosyası Yükleyin",
                                file_types=[".pdf"],
                                type="filepath"
                            )
                            
                            gr.Markdown("### ⚙️ İşlem Ayarları")
                            
                            confidence_threshold = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                label="🎯 Güven Eşiği"
                            )
                            
                            replacement_strategy = gr.Radio(
                                choices=["Maskeleme (*)", "Sahte Değerler", "Genel Değerler"],
                                value="Sahte Değerler",
                                label="🔁 Değiştirme Stratejisi"
                            )
                            
                            with gr.Accordion("🧩 Özel Değer Haritası (JSON)", open=False):
                                gr.Markdown("Örnek:")
                                example_json = {
                                    "ad_soyad": "Mehmet Yılmaz",
                                    "telefon": "05551234567",
                                    "email": "anonim@example.com",
                                    "adres": "Ankara, Türkiye"
                                }
                                gr.Code(value=json.dumps(example_json, ensure_ascii=False, indent=2), language="json")
                                custom_replacements = gr.Textbox(
                                    label="Özel Değerler (JSON)",
                                    placeholder='{"ad_soyad": "Mehmet Yılmaz", "telefon": "0555..."}',
                                    lines=5
                                )
                            
                            enable_logging = gr.Checkbox(value=True, label="🧾 Log kaydı tut")
                            enable_debug = gr.Checkbox(value=False, label="🐞 Debug bilgilerini göster")
                            
                            process_btn = gr.Button(
                                "🚀 İşlemi Başlat", 
                                variant="primary",
                                interactive=bool(self.ner_pipeline)
                            )
                            
                            if not self.ner_pipeline:
                                gr.Markdown("⚠️ **Uyarı**: Model yüklenemediği için işlem başlatılamaz.")
                        
                        with gr.Column(scale=1):
                            output_pdf = gr.File(label="📤 Anonimleştirilmiş PDF", file_count="single")
                            status_text = gr.Markdown("⏳ Henüz işlem yapılmadı.")
                            summary_md = gr.Markdown("")
                            stats_df = gr.Dataframe(headers=["Veri Türü", "Tespit", "Değiştirildi", "Başarı Oranı", "Örnek"], interactive=False)
                            debug_out = gr.Textbox(label="Debug", lines=15, visible=False)
                    
                        # Buton-click bağlama
                        def _wrap_process(pdf, thr, strat, custom_json, log_on, dbg_on, progress=gr.Progress()):
                            out_path, status, summary, df, dbg = self.process_pdf_with_real_replacement(
                                pdf, thr, strat, custom_json or "", log_on, dbg_on, "font_preserving", progress
                            )
                            # Debug görünürlüğü
                            return (
                                out_path, 
                                status, 
                                summary, 
                                df, 
                                gr.update(value=dbg, visible=bool(dbg_on))
                            )
                        
                        process_btn.click(
                            _wrap_process,
                            inputs=[pdf_input, confidence_threshold, replacement_strategy, custom_replacements, enable_logging, enable_debug],
                            outputs=[output_pdf, status_text, summary_md, stats_df, debug_out],
                            api_name="process_pdf"
                        )
                    
                    # ------------------ Model Test ------------------
                    with gr.TabItem("🧪 Model Test"):
                        gr.Markdown("### Custom NER Model Testi")
                        
                        test_text = gr.Textbox(
                            label="Test Metni",
                            placeholder="Örnek: Ahmet Yılmaz 05551234567 numaralı telefonu kullanıyor.",
                            lines=5
                        )
                        test_confidence = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                            label="Test Güven Eşiği"
                        )
                        test_btn = gr.Button("🔍 Test Et", variant="secondary")
                        test_results = gr.JSON(label="Test Sonuçları")
                        
                        def test_model(text, conf_threshold):
                            if not self.ner_pipeline or not text.strip():
                                return {"error": "Model yüklü değil veya metin boş"}
                            
                            try:
                                results = self.ner_pipeline(text)
                                filtered_results = [
                                    {
                                        'entity': self.map_model_label_to_type(r['entity_group']),
                                        'word': r['word'],
                                        'score': round(r['score'], 3),
                                        'start': r['start'],
                                        'end': r['end'],
                                        'original_label': r['entity_group']
                                    }
                                    for r in results if r['score'] >= conf_threshold
                                ]
                                return {
                                    "entities_found": len(filtered_results),
                                    "results": filtered_results
                                }
                            except Exception as e:
                                return {"error": str(e)}
                        
                        test_btn.click(
                            test_model,
                            inputs=[test_text, test_confidence],
                            outputs=test_results
                        )
                    
                    # ------------------ Geçmiş ------------------
                    with gr.TabItem("🗂️ Geçmiş"):
                        gr.Markdown("Son işlemlerin özeti (loglardan derlenir).")
                        history_df = gr.Dataframe(interactive=False)
                        refresh_btn = gr.Button("🔄 Yenile")
                        
                        def _load_history():
                            return self.get_processing_history()
                        
                        refresh_btn.click(_load_history, inputs=None, outputs=history_df)
                        interface.load(_load_history, inputs=None, outputs=history_df)
                    
                    # ------------------ Yardım ------------------
                    with gr.TabItem("❓ Yardım"):
                        gr.Markdown(f"""
    ### 🤖 Custom NER Model Bilgileri
    - **Model Yolu**: `{self.model_path}`
    - **Model Durumu**: {'✅ Yüklendi ve hazır' if self.ner_pipeline else '❌ Yüklenemedi'}
    - **İşlem Yöntemi**: Font özelliklerini koruyarak değiştirme

            ### 🔧 Gelişmiş Özellikler
    - **Default Font**: Tüm değiştirmeler Helvetica font ile yapılır
    - **Boyut Koruma**: Orijinal font boyutu korunur (uzun metinler için otomatik küçültme)
    - **Renk Koruma**: Orijinal metin rengi korunur
    - **Pozisyon Hassasiyeti**: Metinler tam pozisyonlarında değiştirilir, üst üste binme önlenir
    - **Akıllı Boyutlandırma**: Uzun metinler için font boyutu otomatik ayarlanır
    - **Çoklu Yöntem**: Model + Regex kombinasyonu için maksimum tespit

    ### 📋 Gerekli Paketler
    ```bash
    pip install gradio pandas pymupdf transformers torch faker
    ```

    ### ⚙️ Model Etiket Eşleştirme
    Modelinizin çıkardığı etiketler otomatik olarak uygulama türlerine eşleştirilir:
    - `PERSON/PER` → Ad Soyad
    - `PHONE/PHONE_NUMBER` → Telefon  
    - `EMAIL` → E-posta
    - `ORGANIZATION/ORG` → Şirket
    - `ID_NUMBER/NATIONAL_ID` → TC Kimlik

            ### 🚨 Önemli Notlar
    1. **Yazı Çakışması**: Yeni metinler orijinal pozisyonlarda, default Helvetica font ile yerleştirilir
    2. **Font Boyutu**: Orijinal boyut korunur, uzun metinler için otomatik küçültülür
    3. **Renk Koruma**: Orijinal metin rengi korunur
    4. **Sıralama**: Değiştirmeler sondan başa yapılır (pozisyon kayması önlenir)
                        """)

                return interface

if __name__ == "__main__":
    app = EnhancedAnonymizationApp()
    demo = app.create_enhanced_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)