import gradio as gr
import os
import json
import pandas as pd
from datetime import datetime
import shutil
from typing import List, Dict, Tuple, Optional
import logging
import random
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import fitz  # PyMuPDF
import re
import numpy as np
import unicodedata
# Import custom data lists
from datalar import (
    iban_samples, email_samples, adres_len_samples, ad_soyad_len_samples,
    sirket_len_samples, tarih_samples, telefon_samples, adres_samples,
    para_samples, ad_soyad_samples, sirket_samples
)


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

        # Aynı kelime -> aynı değiştirme mapping'i
        self.replacement_cache = {}

        # Custom data lists - organize by length for exact matching
        self.organized_data = self._organize_data_by_length()

        # Dizinleri oluştur
        for dir_name in ["uploads", "outputs"]:
            os.makedirs(dir_name, exist_ok=True)

    def _normalize_for_pdf_search(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
        s = re.sub(r"[ \t]+", " ", s).strip()
        return s

    def _candidate_queries(self, orig: str) -> List[str]:
        base = self._normalize_for_pdf_search(orig or "")
        cand = {base}

        # boşluk / noktalama normalizasyonları
        cand.add(re.sub(r"\s*([./-])\s*", r"\1", base))
        cand.add(re.sub(r"\s+", " ", base))
        if len(base) <= 64:
            cand.add(base.replace(" ", ""))

        # CASE varyantları
        cand.add(base.upper())
        cand.add(base.lower())
        cand.add(base.title())

        return [c for c in cand if c]

    def _organize_data_by_length(self):
        """Veri listelerini karakter uzunluğuna göre organize et"""
        organized = {
            'ad_soyad': {},
            'telefon': {},
            'email': {},
            'adres': {},
            'sirket': {},
            'iban': {},
            'tarih': {},
            'para': {}
        }

        # Ad Soyad
        for item in ad_soyad_samples:
            length = len(item)
            if length not in organized['ad_soyad']:
                organized['ad_soyad'][length] = []
            organized['ad_soyad'][length].append(item)

        for length_data in ad_soyad_len_samples:
            length = length_data['length']
            if length not in organized['ad_soyad']:
                organized['ad_soyad'][length] = []
            organized['ad_soyad'][length].extend(length_data['samples'])

        # Telefon
        for item in telefon_samples:
            length = len(item)
            if length not in organized['telefon']:
                organized['telefon'][length] = []
            organized['telefon'][length].append(item)

        # Email
        for item in email_samples:
            length = len(item)
            if length not in organized['email']:
                organized['email'][length] = []
            organized['email'][length].append(item)

        # Adres
        for item in adres_samples:
            length = len(item)
            if length not in organized['adres']:
                organized['adres'][length] = []
            organized['adres'][length].append(item)

        for length_data in adres_len_samples:
            length = length_data['length']
            if length not in organized['adres']:
                organized['adres'][length] = []
            organized['adres'][length].extend(length_data['samples'])

        # Şirket
        for item in sirket_samples:
            length = len(item)
            if length not in organized['sirket']:
                organized['sirket'][length] = []
            organized['sirket'][length].append(item)

        for length_data in sirket_len_samples:
            length = length_data['length']
            if length not in organized['sirket']:
                organized['sirket'][length] = []
            organized['sirket'][length].extend(length_data['samples'])

        # IBAN
        for item in iban_samples:
            length = len(item)
            if length not in organized['iban']:
                organized['iban'][length] = []
            organized['iban'][length].append(item)

        # Tarih
        for item in tarih_samples:
            length = len(item)
            if length not in organized['tarih']:
                organized['tarih'][length] = []
            organized['tarih'][length].append(item)

        # Para
        for item in para_samples:
            length = len(item)
            if length not in organized['para']:
                organized['para'][length] = []
            organized['para'][length].append(item)

        return organized
    
    def _get_deterministic_random(self, seed_text: str) -> random.Random:
        """Deterministik random generator"""
        seeded_value = abs(hash(seed_text)) % (10 ** 9)
        rng = random.Random()
        rng.seed(seeded_value)
        return rng

    def generate_text_with_exact_length(self, target_length: int, entity_type: str, seed_text: str) -> str:
        """Custom Lists'ten TAM uzunluk eşleşmesi ile değer döndür"""
        rng = self._get_deterministic_random(seed_text)

        lengths_dict = self.organized_data.get(entity_type, {})
        samples = lengths_dict.get(target_length, [])

        if samples:
            return rng.choice(samples)

        raise KeyError(f"No sample in lists for entity_type='{entity_type}' with length={target_length}")
    
    def _search_quads_near(self, page, query: str, ref_bbox, max_hits=64):
        try:
            tp = page.get_textpage()

            flags = 0
            for name in ("TEXT_SEARCH_IGNORE_CASE", "TEXT_IGNORECASE"):
                if hasattr(fitz, name):
                    flags |= getattr(fitz, name)

            try:
                hits = tp.search(query, hit_max=max_hits, quads=True, flags=flags)
            except TypeError:
                hits = tp.search(query, hit_max=max_hits, quads=True)

            if not hits:
                return None

            rx0, ry0, rx1, ry1 = ref_bbox
            best, best_dist = None, float("inf")
            for h in hits:
                q = h[0] if isinstance(h, (list, tuple)) else h
                rect = fitz.Rect(q.rect) if hasattr(q, "rect") else fitz.Rect(q)
                dist = abs(rect.x0 - rx0) + abs(rect.y0 - ry0)
                if dist < best_dist:
                    best, best_dist = h, dist
            return best
        except Exception:
            return None

    def _rect_from_block_slice_chars(self, page, text_block_info) -> Optional[fitz.Rect]:
        """Char bbox'larını birleştirerek Rect döndürür"""
        try:
            block_text = text_block_info.get('block_text', '')
            rel_s = int(text_block_info.get('relative_start', 0))
            rel_e = int(text_block_info.get('relative_end', 0))
            if not block_text or rel_e <= rel_s:
                return None

            target_text_norm = self._normalize_for_pdf_search(block_text)
            bx0, by0, bx1, by1 = text_block_info.get('bbox', (0,0,0,0))

            raw = page.get_text("rawdict")
            for b in raw.get("blocks", []):
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        span_text = (s.get("text") or "")
                        span_bbox = s.get("bbox", None)
                        if not span_bbox:
                            continue

                        span_text_norm = self._normalize_for_pdf_search(span_text)
                        sx0, sy0, sx1, sy1 = span_bbox
                        bbox_dist = abs(sx0 - bx0) + abs(sy0 - by0)

                        if span_text_norm == target_text_norm and bbox_dist < 10.0:
                            chars = s.get("chars")
                            if not chars:
                                total_w = sx1 - sx0
                                if total_w <= 0:
                                    return fitz.Rect(span_bbox)
                                frac_s = rel_s / max(len(span_text), 1)
                                frac_e = rel_e / max(len(span_text), 1)
                                x0 = sx0 + total_w * frac_s
                                x1 = sx0 + total_w * frac_e
                                return fitz.Rect(min(x0,x1), sy0, max(x0,x1), sy1)

                            xs, ys = [], []
                            N = len(chars)
                            a = max(0, min(rel_s, N-1))
                            bnd = max(a+1, min(rel_e, N))
                            for ch in chars[a:bnd]:
                                cb = ch.get("bbox")
                                if not cb:
                                    continue
                                cx0, cy0, cx1, cy1 = cb
                                xs.extend([cx0, cx1]); ys.extend([cy0, cy1])

                            if xs and ys:
                                return fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                            return fitz.Rect(span_bbox)
            return None
        except Exception:
            return None

    def _rects_from_hit(self, hit) -> List[fitz.Rect]:
        rects = []
        def to_rect(obj):
            if isinstance(obj, fitz.Rect):
                return obj
            if hasattr(obj, "rect"):
                try: return fitz.Rect(obj.rect)
                except Exception: pass
            if isinstance(obj, (list, tuple)) and len(obj) == 4 and all(hasattr(p, "x") and hasattr(p, "y") for p in obj):
                xs = [p.x for p in obj]; ys = [p.y for p in obj]
                return fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            return None
        def walk(o):
            r = to_rect(o)
            if r is not None:
                rects.append(r); return
            if isinstance(o, (list, tuple)):
                for it in o: walk(it)
        walk(hit)
        return [fitz.Rect(r) for r in rects if r is not None]

    def _generate_valid_tc_with_length(self, target_length: int, rng: random.Random) -> str:
        """Geçerli TC kimlik numarası üret"""
        if target_length != 11:
            return ''.join(rng.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=target_length))

        # 11 karakter için geçerli TC üret
        for _ in range(100):
            d = [0] * 11
            d[0] = rng.randint(1, 9)
            for i in range(1, 9):
                d[i] = rng.randint(0, 9)

            odd_sum = sum(d[i] for i in range(0, 9, 2))
            even_sum = sum(d[i] for i in range(1, 8, 2))
            d[9] = (odd_sum * 7 - even_sum) % 10
            d[10] = sum(d[:10]) % 10

            tc = "".join(map(str, d))
            if self.validate_turkish_id(tc):
                return tc

        return '12345678901'

    def load_custom_ner_model(self):
        """Özel NER modelini yükle"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model bulunamadı: {self.model_path}")
                return None

            self.logger.info(f"Model yükleniyor: {self.model_path}")

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path)

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
                                    bbox_raw = span.get("bbox", (0, 0, 0, 0))
                                    if hasattr(bbox_raw, "__iter__"):
                                        bbox = tuple(float(x) for x in bbox_raw)
                                    else:
                                        try:
                                            bbox = (float(bbox_raw.x0), float(bbox_raw.y0),
                                                    float(bbox_raw.x1), float(bbox_raw.y1))
                                        except Exception:
                                            bbox = (0.0, 0.0, 0.0, 0.0)

                                    text_blocks.append({
                                        'page': int(page_num),
                                        'text': text,
                                        'bbox': bbox,
                                        'font': str(span.get("font") or "Unknown"),
                                        'size': float(span.get("size", 12)),
                                        'flags': int(span.get("flags", 0)),
                                        'color': int(span.get("color", 0)) if isinstance(span.get("color", 0), (int, float)) else 0,
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

    def process_pdf_with_real_replacement(self, pdf_file, confidence_threshold: float, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """PDF işleme - GERÇEK değiştirme ile"""
        if pdf_file is None:
            return None, "❌ Lütfen bir PDF dosyası yükleyin."

        if self.ner_pipeline is None:
            return None, "❌ NER modeli yüklenemedi. Model yolunu kontrol edin."

        # Her yeni işlemde cache'i temizle
        self.replacement_cache.clear()

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

            # PDF'ten metin ve pozisyonları çıkar
            text_blocks = self.extract_text_with_positions(input_path)
            full_text = " ".join([block['text'] for block in text_blocks])

            if not full_text.strip():
                return None, "⚠️ PDF'den metin çıkarılamadı."

            progress(0.2, desc="NER analizi yapılıyor...")

            # Custom NER model ile analiz
            entities_detected = self.extract_entities_with_custom_model(full_text, text_blocks, confidence_threshold, progress)

            if not entities_detected:
                return None, "⚠️ PDF'de kişisel bilgi tespit edilmedi."

            progress(0.4, desc="Değiştirme stratejisi uygulanıyor...")

            # Tutarlı değiştirme uygula
            processed_entities = self.apply_replacement_strategy_consistent(entities_detected)

            progress(0.5, desc="PDF'te değişiklikler uygulanıyor...")

            # Font korumalı PDF değiştirme
            success = self.perform_font_preserving_replacement(input_path, processed_entities, output_path, text_blocks, progress)

            if not success:
                return None, "❌ PDF değiştirme işlemi başarısız oldu."

            progress(1.0, desc="Tamamlandı!")

            # Basit başarı mesajı
            status_msg = f"✅ İşlem tamamlandı! {len(processed_entities)} değiştirme yapıldı."
            
            return output_path, status_msg

        except Exception as e:
            self.logger.error(f"PDF işleme hatası: {e}", exc_info=True)
            return None, f"❌ Kritik hata: {str(e)}"

    def process_pdf_with_censoring(self, pdf_file, confidence_threshold: float, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """PDF işleme - SANSÜRLEME ile (**** karakterleri)"""
        if pdf_file is None:
            return None, "❌ Lütfen bir PDF dosyası yükleyin."

        if self.ner_pipeline is None:
            return None, "❌ NER modeli yüklenemedi. Model yolunu kontrol edin."

        # Her yeni işlemde cache'i temizle
        self.replacement_cache.clear()

        try:
            progress(0.05, desc="Dosya hazırlanıyor...")

            # Dosya yolları
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_in = os.path.basename(pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file))
            input_filename = f"input_{timestamp}_{base_in}"
            output_filename = f"censored_{timestamp}_{base_in}"

            input_path = os.path.join("uploads", input_filename)
            output_path = os.path.join("outputs", output_filename)

            # Yüklenen dosyayı kopyala
            shutil.copy2(pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file), input_path)

            progress(0.1, desc="PDF analiz ediliyor...")

            # PDF'ten metin ve pozisyonları çıkar
            text_blocks = self.extract_text_with_positions(input_path)
            full_text = " ".join([block['text'] for block in text_blocks])

            if not full_text.strip():
                return None, "⚠️ PDF'den metin çıkarılamadı."

            progress(0.2, desc="NER analizi yapılıyor...")

            # Custom NER model ile analiz
            entities_detected = self.extract_entities_with_custom_model(full_text, text_blocks, confidence_threshold, progress)

            if not entities_detected:
                return None, "⚠️ PDF'de kişisel bilgi tespit edilmedi."

            progress(0.4, desc="Sansürleme stratejisi uygulanıyor...")

            # Sansürleme uygula
            processed_entities = self.apply_censoring_strategy(entities_detected)

            progress(0.5, desc="PDF'te sansürlemeler uygulanıyor...")

            # Font korumalı PDF sansürleme
            success = self.perform_font_preserving_replacement(input_path, processed_entities, output_path, text_blocks, progress)

            if not success:
                return None, "❌ PDF sansürleme işlemi başarısız oldu."

            progress(1.0, desc="Tamamlandı!")

            status_msg = f"✅ Sansürleme tamamlandı! {len(processed_entities)} kişisel bilgi sansürlendi."
            
            return output_path, status_msg

        except Exception as e:
            self.logger.error(f"PDF sansürleme hatası: {e}", exc_info=True)
            return None, f"❌ Kritik hata: {str(e)}"

    def apply_censoring_strategy(self, entities: List[Dict]) -> List[Dict]:
        """Sansürleme stratejisini uygula - metni * karakterleri ile değiştir"""
        processed = []
        for entity in entities:
            e = entity.copy()
            original = e.get('word', '').strip()
            
            # Orijinal metnin uzunluğu kadar * karakteri oluştur
            censored_text = '*' * len(original)
            e['replacement'] = censored_text
            processed.append(e)

        return processed

    def extract_entities_with_custom_model(self, full_text: str, text_blocks: List[Dict], confidence_threshold: float, progress) -> List[Dict]:
        """Custom NER model ile entity tespiti + TC kimlik için regex"""
        try:
            # Metni parçalara böl
            max_length = 512
            text_chunks = []
            chunk_start = 0

            while chunk_start < len(full_text):
                chunk_end = min(chunk_start + max_length, len(full_text))

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
                            entity_start = chunk['start_offset'] + result['start']
                            entity_end = chunk['start_offset'] + result['end']

                            text_block_info = self.find_text_block_for_position(entity_start, entity_end, text_blocks, full_text)

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

            progress(0.3, desc="TC kimlik regex kontrolü...")

            # TC kimlik için regex tespiti
            tc_entities = self.detect_tc_kimlik_with_blocks(full_text, text_blocks)

            # Birleştir ve temizle
            combined_entities = self.merge_and_clean_entities(all_entities + tc_entities, confidence_threshold)

            self.logger.info(f"Toplam {len(combined_entities)} entity tespit edildi")
            return combined_entities

        except Exception as e:
            self.logger.error(f"Custom NER analiz hatası: {e}")
            return []

    def detect_tc_kimlik_with_blocks(self, full_text: str, text_blocks: List[Dict]) -> List[Dict]:
        """TC Kimlik için regex tespiti"""
        tc_entities = []
        tc_pattern = r'\b[1-9][0-9]{9}[02468]\b'
        
        for match in re.finditer(tc_pattern, full_text):
            tc_no = match.group()
            if self.validate_turkish_id(tc_no):
                text_block_info = self.find_text_block_for_position(match.start(), match.end(), text_blocks, full_text)

                tc_entities.append({
                    'entity': 'tc_kimlik',
                    'word': tc_no,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.95,
                    'method': 'regex_validated',
                    'text_block_info': text_block_info
                })

        return tc_entities

    def map_model_label_to_type(self, model_label: str) -> str:
        """Model etiketlerini uygulama türlerine eşle"""
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

    def find_text_block_for_position(self, start_pos: int, end_pos: int, text_blocks: List[Dict], full_text: str) -> Dict:
        """Verilen pozisyon için text block bilgilerini bul"""
        current_pos = 0

        for block in text_blocks:
            block_text = block['text']
            block_start = current_pos
            block_end = current_pos + len(block_text)

            if start_pos >= block_start and end_pos <= block_end + 1:
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

            current_pos = block_end + 1

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

    def perform_font_preserving_replacement(self, input_path: str, entities: List[Dict], output_path: str, text_blocks: List[Dict], progress) -> bool:
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

                progress(0.6 + (page_num / len(doc)) * 0.2, desc=f"Sayfa {page_num + 1}/{len(doc)} işleniyor...")

                # Entity'leri pozisyona göre sırala (tersten)
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
            original_text = (entity.get('word') or '').strip()
            replacement_text = (entity.get('replacement') or original_text).strip()
            if not original_text or replacement_text == original_text:
                return False

            text_block_info = entity.get('text_block_info') or {}
            bbox = text_block_info.get('bbox')
            if not bbox:
                return False

            block_text = text_block_info.get('block_text', '')
            rel_s = int(text_block_info.get('relative_start', 0))
            rel_e = int(text_block_info.get('relative_end', 0))
            block_slice = block_text[rel_s:rel_e] if (0 <= rel_s <= len(block_text) and 0 <= rel_e <= len(block_text)) else ''
            search_text = (block_slice or original_text).strip()

            # Aday sorgular
            candidates = self._candidate_queries(search_text) or [search_text]

            # Quad arama
            hit_quads = None
            for q in candidates:
                hit_quads = self._search_quads_near(page, q, bbox)
                if hit_quads:
                    break

            # Fallback
            if not hit_quads:
                char_rect = self._rect_from_block_slice_chars(page, text_block_info)
                if char_rect:
                    rects = [char_rect]
                else:
                    return False
            else:
                rects = self._rects_from_hit(hit_quads)

            if not rects:
                return False

            # Redaksiyon
            for r in rects:
                bg = self._sample_background_color(page, r, margin=1.0, ring=8)
                page.add_redact_annot(r, fill=bg)
            page.apply_redactions() 

            # Yazı yerleştirme
            first_rect = rects[0]
            font_size = float(text_block_info.get('size', 12.0))
            font_color = text_block_info.get('color', 0)

            # Renk dönüştürme
            if isinstance(font_color, int):
                rgb_color = (
                    ((font_color >> 16) & 255) / 255.0,
                    ((font_color >> 8) & 255) / 255.0,
                    (font_color & 255) / 255.0
                )
            else:
                rgb_color = (0, 0, 0)

            # Font boyutu ayarlama
            rect_width = first_rect.width
            text_width = fitz.get_text_length(replacement_text, fontname='helv', fontsize=font_size)
            if text_width > rect_width * 1.1:
                font_size = max(font_size * (rect_width / max(text_width, 1e-6)) * 0.9, 6)

            insert_point = (first_rect.x0, first_rect.y1 - 2)
            page.insert_text(
                insert_point,
                replacement_text,
                fontsize=font_size,
                fontname='helv',
                color=rgb_color,
                render_mode=0
            )

            return True

        except Exception as e:
            self.logger.error(f"Entity değiştirme hatası: {e}")
            return False
        
    def _sample_background_color(self, page, rect, margin=1.5, ring=4) -> tuple:
        """Dikdörtgenin çevresinden arka plan rengi örnekler"""
        try:
            r = fitz.Rect(rect)
            outer = fitz.Rect(r).inflate(margin + ring)
            inner = fitz.Rect(r).inflate(margin)

            pm = page.get_pixmap(clip=outer, alpha=False)
            w, h, n = pm.width, pm.height, pm.n

            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(h, w, n)
            rgb = arr[:, :, :3] if n >= 3 else np.repeat(arr, 3, axis=2)

            sx = w / max(outer.width, 1e-6)
            sy = h / max(outer.height, 1e-6)
            ix0 = max(0, min(w, int((inner.x0 - outer.x0) * sx)))
            iy0 = max(0, min(h, int((inner.y0 - outer.y0) * sy)))
            ix1 = max(0, min(w, int((inner.x1 - outer.x1) * sx)))
            iy1 = max(0, min(h, int((inner.y1 - outer.y0) * sy)))

            mask = np.ones((h, w), dtype=bool)
            mask[iy0:iy1, ix0:ix1] = False

            samples = rgb[mask]
            if samples.size == 0:
                samples = rgb.reshape(-1, 3)

            # Çok koyu pikselleri filtrele
            lum = 0.2126 * samples[:, 0] + 0.7152 * samples[:, 1] + 0.0722 * samples[:, 2]
            use = lum > 60
            if np.sum(use) >= 50:
                samples = samples[use]

            med = np.median(samples, axis=0)
            return (float(med[0] / 255.0), float(med[1] / 255.0), float(med[2] / 255.0))
        except Exception:
            return (0.96, 0.96, 0.96)

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

    def generate_consistent_replacement(self, original_text: str, entity_type: str) -> str:
        """Tutarlı değiştirme üretir"""
        cache_key = f"{original_text}_{entity_type}"
        if cache_key in self.replacement_cache:
            return self.replacement_cache[cache_key]

        target_length = len(original_text)

        try:
            replacement = self.generate_text_with_exact_length(target_length, entity_type, original_text)
        except KeyError:
            # Liste yoksa orijinal bırak
            replacement = original_text

        # Değişiklik olduysa cache'le
        if replacement != original_text:
            self.replacement_cache[cache_key] = replacement
        return replacement

    def apply_replacement_strategy_consistent(self, entities: List[Dict]) -> List[Dict]:
        """Tutarlı değiştirme stratejisini uygula"""
        processed = []
        for entity in entities:
            e = entity.copy()
            original = e.get('word', '').strip()
            entity_type = e.get('entity', '').strip()

            replacement = self.generate_consistent_replacement(original, entity_type)
            e['replacement'] = replacement
            processed.append(e)

        return processed

    def create_enhanced_interface(self):
        """İki Sekmeli Gradio arayüzü - Değiştirme ve Sansürleme"""
        css = """
        .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .tab-nav { margin-bottom: 20px; }
        """

        try:
            theme = gr.themes.Soft()
        except Exception:
            theme = None

        with gr.Blocks(css=css, title="PDF Kişisel Bilgi Anonimleştirici - Gelişmiş", theme=theme) as interface:
            gr.Markdown("""
            # 🔐 PDF Kişisel Bilgi Anonimleştirici — Gelişmiş Sürüm
            **İki farklı yöntem ile PDF'lerdeki kişisel bilgileri anonimleştirin**
            """)

            with gr.Tabs():
                # SEKME 1: Değiştirme (Custom Lists ile)
                with gr.Tab("🔄 Değiştirme (Custom Lists)", elem_classes="tab-nav"):
                    gr.Markdown("""
                    ### Custom Lists ile Değiştirme
                    Kişisel bilgiler, uzunluk korunarak benzer gerçekçi verilerle değiştirilir.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_input_replace = gr.File(
                                label="📎 PDF Dosyası Yükleyin",
                                file_types=[".pdf"],
                                type="filepath"
                            )

                            confidence_threshold_replace = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                label="🎯 Güven Eşiği"
                            )

                            process_btn_replace = gr.Button(
                                "🚀 Değiştirme İşlemini Başlat",
                                variant="primary",
                                interactive=bool(self.ner_pipeline)
                            )

                            if not self.ner_pipeline:
                                gr.Markdown("⚠️ **Uyarı**: Model yüklenemediği için işlem başlatılamaz.")

                        with gr.Column(scale=1):
                            output_pdf_replace = gr.File(label="📤 Değiştirilmiş PDF", file_count="single")
                            status_text_replace = gr.Markdown("")

                # SEKME 2: Sansürleme (* karakterleri ile)
                with gr.Tab("🚫 Sansürleme (Yıldız ile)", elem_classes="tab-nav"):
                    gr.Markdown("""
                    ### Yıldız (*) Karakterleri ile Sansürleme
                    Kişisel bilgiler tespit edilip, karakter sayısı kadar * ile değiştirilir.
                    **Örnek:** "Ahmet Yılmaz" → "*****\u00A0******"
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_input_censor = gr.File(
                                label="📎 PDF Dosyası Yükleyin",
                                file_types=[".pdf"],
                                type="filepath"
                            )

                            confidence_threshold_censor = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                label="🎯 Güven Eşiği"
                            )

                            process_btn_censor = gr.Button(
                                "🚫 Sansürleme İşlemini Başlat",
                                variant="secondary",
                                interactive=bool(self.ner_pipeline)
                            )

                            if not self.ner_pipeline:
                                gr.Markdown("⚠️ **Uyarı**: Model yüklenemediği için işlem başlatılamaz.")

                        with gr.Column(scale=1):
                            output_pdf_censor = gr.File(label="📤 Sansürlenmiş PDF", file_count="single")
                            status_text_censor = gr.Markdown("")

            # OLAY YÖNETİCİLERİ

            def _run_replacement(pdf, thr, progress=gr.Progress()):
                out_path, status = self.process_pdf_with_real_replacement(
                    pdf_file=pdf,
                    confidence_threshold=thr,
                    progress=progress
                )
                return out_path, (status or "")

            def _run_censoring(pdf, thr, progress=gr.Progress()):
                out_path, status = self.process_pdf_with_censoring(
                    pdf_file=pdf,
                    confidence_threshold=thr,
                    progress=progress
                )
                return out_path, (status or "")

            process_btn_replace.click(
                _run_replacement,
                inputs=[pdf_input_replace, confidence_threshold_replace],
                outputs=[output_pdf_replace, status_text_replace],
                api_name="process_pdf_replacement"
            )

            process_btn_censor.click(
                _run_censoring,
                inputs=[pdf_input_censor, confidence_threshold_censor],
                outputs=[output_pdf_censor, status_text_censor],
                api_name="process_pdf_censoring"
            )

        return interface


if __name__ == "__main__":
    app = EnhancedAnonymizationApp()
    demo = app.create_enhanced_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)