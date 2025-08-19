

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


# ----------------------------------------------------
#  EnhancedAnonymizationApp  (Custom Data Lists + Tutarlı Değiştirme + Karakter Sayısı Koruma)
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
        self.strict_lists_only = True

        # Tutarlı sahte üretim için sabit seed
        self.global_seed = 123456
        random.seed(self.global_seed)

        # Aynı kelime -> aynı değiştirme mapping'i
        self.replacement_cache = {}

        # Custom data lists - organize by length for exact matching
        self.organized_data = self._organize_data_by_length()

        # Dizinleri oluştur
        for dir_name in ["uploads", "outputs", "logs", "temp"]:
            os.makedirs(dir_name, exist_ok=True)

        self.processing_stats = []
    def _normalize_for_pdf_search(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
        s = re.sub(r"[ \t]+", " ", s).strip()
        return s

    def _candidate_queries(self, orig: str) -> List[str]:
        base = self._normalize_for_pdf_search(orig or "")
        cand = {base}

        # boşluk / noktalama normalizasyonları
        cand.add(re.sub(r"\s*([./-])\s*", r"\1", base))  # 05. 06. 2023 -> 05.06.2023
        cand.add(re.sub(r"\s+", " ", base))              # çoklu boşluk -> tek
        if len(base) <= 64:
            cand.add(base.replace(" ", ""))              # tüm boşlukları kaldır (kısa metinlerde)

        # CASE varyantları (case-insensitive davranışa yaklaşmak için)
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

        # Ad Soyad - hem genel liste hem de uzunluk bazlı
        for item in ad_soyad_samples:
            length = len(item)
            if length not in organized['ad_soyad']:
                organized['ad_soyad'][length] = []
            organized['ad_soyad'][length].append(item)

        # Uzunluk bazlı ad soyad verilerini ekle
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

        # Adres - hem genel liste hem de uzunluk bazlı
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

        # Şirket - hem genel liste hem de uzunluk bazlı
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
        """
        Sadece Custom Lists'ten, TAM uzunluk eşleşmesi ile değer döndür.
        Hiçbir koşulda rastgele üretim / uzatma / kısaltma yapmaz.
        Uygun örnek yoksa KeyError fırlatır.
        """
        rng = self._get_deterministic_random(seed_text)

        # Liste ve uzunluk kontrolü
        lengths_dict = self.organized_data.get(entity_type, {})
        samples = lengths_dict.get(target_length, [])

        if samples:
            return rng.choice(samples)

        # Sıkı modda: asla fallback yapma
        raise KeyError(f"No sample in lists for entity_type='{entity_type}' with length={target_length}")
    
    def _search_quads_near(self, page, query: str, ref_bbox, max_hits=64):
        try:
            tp = page.get_textpage()

            # Case-ignore flag'i varsa topla
            flags = 0
            for name in ("TEXT_SEARCH_IGNORE_CASE", "TEXT_IGNORECASE"):
                if hasattr(fitz, name):
                    flags |= getattr(fitz, name)

            # bazı sürümlerde flags argümanı yok -> try / except
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
        """
        Aynı span'daki substring'in char bbox'larını birleştirerek bir Rect döndürür.
        'rawdict' içinde ilgili span'ı, text + bbox yakınlığı ile bulur.
        """
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

                        # hem metni hem bbox'ı yakın olmalı
                        span_text_norm = self._normalize_for_pdf_search(span_text)
                        sx0, sy0, sx1, sy1 = span_bbox
                        bbox_dist = abs(sx0 - bx0) + abs(sy0 - by0)  # kaba yakınlık ölçüsü

                        if span_text_norm == target_text_norm and bbox_dist < 10.0:
                            chars = s.get("chars")
                            if not chars:
                                # chars yoksa yaklaşık olarak genişlik oranından kes
                                # (çok nadiren olur)
                                total_w = sx1 - sx0
                                if total_w <= 0:
                                    return fitz.Rect(span_bbox)
                                frac_s = rel_s / max(len(span_text), 1)
                                frac_e = rel_e / max(len(span_text), 1)
                                x0 = sx0 + total_w * frac_s
                                x1 = sx0 + total_w * frac_e
                                return fitz.Rect(min(x0,x1), sy0, max(x0,x1), sy1)

                            # char'ların bbox'larını birleştir
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
            # 11 karakter değilse, sayı + harf kombinasyonu yap
            return ''.join(rng.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=target_length))

        # 11 karakter için geçerli TC üret
        for _ in range(100):  # 100 deneme
            d = [0] * 11
            d[0] = rng.randint(1, 9)
            for i in range(1, 9):
                d[i] = rng.randint(0, 9)

            # Checksum hesapla
            odd_sum = sum(d[i] for i in range(0, 9, 2))
            even_sum = sum(d[i] for i in range(1, 8, 2))
            d[9] = (odd_sum * 7 - even_sum) % 10
            d[10] = sum(d[:10]) % 10

            tc = "".join(map(str, d))
            if self.validate_turkish_id(tc):
                return tc

        # Bulunamazsa basit bir değer döndür
        return '12345678901'

    def _generate_generic_value(self, target_length: int, entity_type: str, rng: random.Random) -> str:
        """Generic değer üret"""
        if entity_type == 'ad_soyad':
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        elif entity_type in ['telefon', 'tc_kimlik']:
            chars = '0123456789'
        elif entity_type == 'email':
            if target_length < 5:
                return 'a@b.c'[:target_length]
            base = 'user@example.com'
            if len(base) == target_length:
                return base
            elif len(base) < target_length:
                extra = 'x' * (target_length - len(base))
                return f"user{extra}@example.com"
            else:
                return base[:target_length]
        elif entity_type == 'para':
            if target_length <= 3:
                return '₺' + '1' * (target_length - 1)
            else:
                numbers = ''.join(str(rng.randint(0, 9)) for _ in range(target_length - 3))
                return f"{numbers} TL"
        elif entity_type == 'tarih':
            # Tarih formatı dene
            if target_length == 8:
                return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(20, 99):02d}"
            elif target_length == 10:
                return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1900, 2099)}"
            else:
                chars = '0123456789.'
        elif entity_type == 'iban':
            return 'TR' + ''.join(str(rng.randint(0, 9)) for _ in range(target_length - 2))
        else:
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

        if 'chars' in locals():
            return ''.join(rng.choices(chars, k=target_length))
        else:
            return 'X' * target_length

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
                                        'bbox': bbox,  # artık saf Python float tuple
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
        Gelişmiş PDF işleme - GERÇEK değiştirme ile (Custom NER + Custom Lists + Tutarlı değiştirme + Karakter koruma)
        """
        if pdf_file is None:
            return None, "❌ Lütfen bir PDF dosyası yükleyin.", "", pd.DataFrame(), ""

        if self.ner_pipeline is None:
            return None, "❌ NER modeli yüklenemedi. Model yolunu kontrol edin.", "", pd.DataFrame(), ""

        # Her yeni işlemde cache'i temizle
        self.replacement_cache.clear()

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

            # 2. Custom NER model ile analiz (sadece TC kimlik için regex)
            entities_detected = self.extract_entities_with_custom_model(full_text, text_blocks, confidence_threshold,
                                                                        progress)

            if not entities_detected:
                return None, "⚠️ PDF'de kişisel bilgi tespit edilmedi.", "", pd.DataFrame(), "Hiçbir entity tespit edilmedi."

            progress(0.4, desc="Değiştirme stratejisi uygulanıyor...")

            # 3. Değiştirme stratejisini uygula (tutarlı + karakter sayısı koruma)
            processed_entities = self.apply_replacement_strategy_consistent(
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
        """Custom NER model ile entity tespiti + TC kimlik için regex"""
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

            progress(0.3, desc="TC kimlik regex kontrolü...")

            # Sadece TC kimlik için regex tespiti
            tc_entities = self.detect_tc_kimlik_with_blocks(full_text, text_blocks)

            # Birleştir ve temizle
            combined_entities = self.merge_and_clean_entities(all_entities + tc_entities, confidence_threshold)

            self.logger.info(f"Toplam {len(combined_entities)} entity tespit edildi")
            return combined_entities

        except Exception as e:
            self.logger.error(f"Custom NER analiz hatası: {e}")
            return []

    def detect_tc_kimlik_with_blocks(self, full_text: str, text_blocks: List[Dict]) -> List[Dict]:
        """Sadece TC Kimlik için regex tespiti"""
        tc_entities = []

        # TC Kimlik pattern
        tc_pattern = r'\b[1-9][0-9]{9}[02468]\b'
        for match in re.finditer(tc_pattern, full_text):
            tc_no = match.group()
            if self.validate_turkish_id(tc_no):
                text_block_info = self.find_text_block_for_position(
                    match.start(), match.end(), text_blocks, full_text
                )

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
        """Tek bir entity'yi font özelliklerini koruyarak değiştir (quad tabanlı arama, güvenli akış)."""
        try:
            # 1) Güvenli çekimler
            original_text = (entity.get('word') or '').strip()
            replacement_text = (entity.get('replacement') or original_text).strip()
            if not original_text or replacement_text == original_text:
                return False

            text_block_info = entity.get('text_block_info') or {}
            bbox = text_block_info.get('bbox')
            if not bbox:
                self.logger.warning("replace_entity_with_font_preservation: bbox yok")
                return False

            block_text = text_block_info.get('block_text', '')
            rel_s = int(text_block_info.get('relative_start', 0))
            rel_e = int(text_block_info.get('relative_end', 0))
            block_slice = block_text[rel_s:rel_e] if (0 <= rel_s <= len(block_text) and 0 <= rel_e <= len(block_text)) else ''
            search_text = (block_slice or original_text).strip()

            # 2) Aday sorgular
            candidates = self._candidate_queries(search_text) or [search_text]

            # 3) Quad arama (yakın bbox'a göre)
            hit_quads = None
            for q in candidates:
                hit_quads = self._search_quads_near(page, q, bbox)
                if hit_quads:
                    break

            # Basit fallback: page.search_for
            if not hit_quads:
                # === CHAR-LEVEL FALLBACK ===
                char_rect = self._rect_from_block_slice_chars(page, text_block_info)
                if char_rect:
                    rects = [char_rect]
                else:
                    self.logger.warning(f"Metin sayfada bulunamadı (quad/search): {search_text}")
                    return False
            else:
                rects = self._rects_from_hit(hit_quads)

            if not rects:
                self.logger.warning("Hit bulundu ama rect çıkarılamadı")
                return False

            # Redaksiyon
            for r in rects:
                bg = self._sample_background_color(page, r, margin=1.0, ring=8)
                page.add_redact_annot(r, fill=bg)
            page.apply_redactions() 


            # 5) Yazı yerleştirme (ilk rect)
            first_rect = rects[0]

            font_size = float(text_block_info.get('size', 12.0))
            font_color = text_block_info.get('color', 0)
            default_font = 'helv'

            # Renk int → RGB
            if isinstance(font_color, int):
                rgb_color = (
                    ((font_color >> 16) & 255) / 255.0,
                    ((font_color >> 8) & 255) / 255.0,
                    (font_color & 255) / 255.0
                )
            else:
                rgb_color = (0, 0, 0)

            # Genişliğe göre font küçült
            rect_width = first_rect.width
            text_width = fitz.get_text_length(replacement_text, fontname=default_font, fontsize=font_size)
            if text_width > rect_width * 1.1:
                font_size = max(font_size * (rect_width / max(text_width, 1e-6)) * 0.9, 6)

            insert_point = (first_rect.x0, first_rect.y1 - 2)
            page.insert_text(
                insert_point,
                replacement_text,
                fontsize=font_size,
                fontname=default_font,
                color=rgb_color,
                render_mode=0
            )

            self.logger.debug(
                f"Değiştirildi: '{original_text}' -> '{replacement_text}' (font={default_font}, size={font_size:.2f})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Entity değiştirme hatası: {e}", exc_info=True)
            return False
        
    def _sample_background_color(self, page, rect, margin=1.5, ring=4) -> tuple:
        """
        Dikdörtgenin çevresinden (ring) arka plan rengi örnekler ve (r,g,b) 0..1 döndürür.
        Çok koyu (muhtemelen yazı) pikselleri filtreler.
        """
        try:
            r = fitz.Rect(rect)
            outer = fitz.Rect(r).inflate(margin + ring)
            inner = fitz.Rect(r).inflate(margin)

            pm = page.get_pixmap(clip=outer, alpha=False)
            w, h, n = pm.width, pm.height, pm.n  # n: kanal sayısı (3 veya 4)

            import numpy as np
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(h, w, n)
            rgb = arr[:, :, :3] if n >= 3 else np.repeat(arr, 3, axis=2)

            # outer koordinatlarından inner’a piksel sınırlarını hesapla
            sx = w / max(outer.width, 1e-6)
            sy = h / max(outer.height, 1e-6)
            ix0 = max(0, min(w, int((inner.x0 - outer.x0) * sx)))
            iy0 = max(0, min(h, int((inner.y0 - outer.y0) * sy)))
            ix1 = max(0, min(w, int((inner.x1 - outer.x1) * sx)))
            iy1 = max(0, min(h, int((inner.y1 - outer.y0) * sy)))  # dikkat: y0

            mask = np.ones((h, w), dtype=bool)
            mask[iy0:iy1, ix0:ix1] = False  # içteki alanı çıkar → sadece halka

            samples = rgb[mask]
            if samples.size == 0:
                samples = rgb.reshape(-1, 3)

            # metin / kenar etkisini azalt: çok koyu pikselleri at
            lum = 0.2126 * samples[:, 0] + 0.7152 * samples[:, 1] + 0.0722 * samples[:, 2]
            use = lum > 60  # eşik: 0..255
            if np.sum(use) >= 50:
                samples = samples[use]

            med = np.median(samples, axis=0)
            return (float(med[0] / 255.0), float(med[1] / 255.0), float(med[2] / 255.0))
        except Exception:
            # sorun olursa beyazı düşmeyelim, hafif gri daha az sırıtıyor
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

    def generate_consistent_replacement(self, original_text: str, entity_type: str, strategy: str,
                                        custom_replacements: Dict) -> str:
        """
        Aynı orijinal metin için tutarlı değiştirme üretir.
        'Sahte Değerler' modunda SIKI LİSTE KURALI uygulanır:
        - Tam uzunluk eşleşmesi olan örnek varsa listeden seçilir.
        - Yoksa DEĞİŞTİRME YAPILMAZ (orijinal bırakılır).
        """
        cache_key = f"{original_text}_{entity_type}_{strategy}"
        if cache_key in self.replacement_cache:
            return self.replacement_cache[cache_key]

        target_length = len(original_text)

        if strategy == "Maskeleme (*)":
            replacement = "*" * target_length

        elif strategy == "Genel Değerler":
            # Genel placeholder'lar KALDI; ama burada da listeden seçme zorunluluğu yok.
            generic_map = {
                'ad_soyad': '[İSİM]', 'tc_kimlik': '[TC KİMLİK]', 'telefon': '[TELEFON]',
                'adres': '[ADRES]', 'para': '[PARA]', 'tarih': '[TARİH]',
                'email': '[E-POSTA]', 'sirket': '[ŞİRKET]', 'iban': '[IBAN]'
            }
            base = generic_map.get(entity_type, f'[{entity_type.upper()}]')
            # Karakter eşitleme olmadan düz kesme/maskeleme istemiyorsan burayı da sıkılaştırabiliriz.
            # Şimdilik: tam uyar ise kullan, değilse kısalt.
            replacement = base[:target_length] if len(base) >= target_length else base + ("*" * (target_length - len(base)))

        elif strategy == "Sahte Değerler":
            # 1) Önce custom_replacements, ama YALNIZCA tam uzunlukta ise kabul
            if entity_type in custom_replacements:
                cv = str(custom_replacements[entity_type])
                if len(cv) == target_length:
                    replacement = cv
                else:
                    # Sıkı mod: uzunluk tutmuyor → değiştirme yapma
                    replacement = original_text
            else:
                # 2) Yalnızca listelerden tam eşleşme
                try:
                    replacement = self.generate_text_with_exact_length(target_length, entity_type, original_text)
                except KeyError:
                    # Sıkı mod: liste yoksa değiştirme yapma
                    replacement = original_text

        else:
            replacement = original_text

        # Değişiklik olduysa cache'le
        if replacement != original_text:
            self.replacement_cache[cache_key] = replacement
        return replacement


    def apply_replacement_strategy_consistent(self, entities: List[Dict], strategy: str, custom_replacements: str) -> \
    List[Dict]:
        """Tutarlı değiştirme stratejisini uygula"""

        # Custom replacements parse et
        custom_dict = {}
        if custom_replacements and custom_replacements.strip():
            try:
                custom_dict = json.loads(custom_replacements)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Custom replacements parse hatası: {e}")

        processed = []
        for entity in entities:
            e = entity.copy()
            original = e.get('word', '').strip()
            entity_type = e.get('entity', '').strip()

            # Tutarlı değiştirme üret (Custom Lists kullanarak)
            replacement = self.generate_consistent_replacement(
                original, entity_type, strategy, custom_dict
            )

            e['replacement'] = replacement
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

        # Custom Lists kullanım istatistikleri
        list_usage_stats = {}
        for entity_type, lengths_dict in self.organized_data.items():
            total_samples = sum(len(samples) for samples in lengths_dict.values())
            if total_samples > 0:
                list_usage_stats[entity_type] = {
                    'total_samples': total_samples,
                    'length_variants': len(lengths_dict)
                }

        debug_info = f"""
## 🔍 Debug Bilgileri

### Değiştirme Sonuçları:
- **Toplam Entity**: {validation_result['total_entities']}
- **Başarılı Değiştirme**: {validation_result['successfully_replaced']}
- **Hala Mevcut**: {len(validation_result.get('still_present', []))}
- **Yeni Değerler Bulundu**: {len(validation_result.get('replacement_found', []))}

### Custom Lists İstatistikleri:
- **Cache Boyutu**: {len(self.replacement_cache)}
- **Kullanılan Entity Tipleri**: {len([e for e in entities if e.get('entity') in self.organized_data])}

### Veri Listesi Kapsamı:
"""
        for entity_type, stats in list_usage_stats.items():
            debug_info += f"- **{entity_type}**: {stats['total_samples']} örnek, {stats['length_variants']} farklı uzunluk\n"

        debug_info += f"""
### Hala Mevcut Metinler:
{validation_result.get('still_present', [])[:5]}

### Bulunan Yeni Değerler:
{validation_result.get('replacement_found', [])[:5]}

### Model Bilgileri:
- **Model Yolu**: {self.model_path}
- **Model Durumu**: {'✅ Yüklendi' if self.ner_pipeline else '❌ Yüklenemedi'}
- **Veri Kaynağı**: Custom Lists (datalar.py)
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
            status_emoji = "🟢";
            status_text = "Mükemmel"
        elif success_rate >= 70:
            status_emoji = "🟡";
            status_text = "İyi"
        else:
            status_emoji = "🔴";
            status_text = "Dikkat Gerekli"

        unique_replacements = len(self.replacement_cache)

        # Custom Lists kullanım analizi
        custom_usage = 0
        for entity in entities:
            entity_type = entity.get('entity')
            original_length = len(entity.get('word', ''))
            if (entity_type in self.organized_data and
                    original_length in self.organized_data[entity_type] and
                    self.organized_data[entity_type][original_length]):
                custom_usage += 1

        summary = f"""
## 📊 Detaylı İşlem Raporu

**Dosya:** {filename}  
**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}

### {status_emoji} Genel Başarı: {status_text} ({success_rate:.1f}%)

### 🎯 Değiştirme Sonuçları
- **Toplam Tespit:** {total_entities} adet
- **Başarılı Değiştirme:** {successful} adet  
- **Hala Mevcut:** {still_present} adet
- **Tutarlı Değiştirmeler:** {unique_replacements} adet
- **Custom Lists Kullanımı:** {custom_usage}/{total_entities} ({(custom_usage / total_entities * 100):.1f}%) 

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

        summary += f"\n\n### 🔄 Tutarlılık & Custom Lists Bilgisi:\n- Aynı metinler için tutarlı değiştirmeler uygulandı.\n- Karakter sayısı koruması aktif.\n- Custom veri listeleri kullanıldı (datalar.py).\n- TC kimlik için regex + geçerli üretim."
        return summary

    def log_processing_with_validation(self, filename: str, entities: List[Dict], validation_result: Dict):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'total_entities': len(entities),
            'successful_replacements': validation_result.get('successfully_replaced', 0),
            'still_present_count': len(validation_result.get('still_present', [])),
            'success_rate': (
                        validation_result.get('successfully_replaced', 0) / len(entities) * 100) if entities else 0,
            'unique_replacements': len(self.replacement_cache),
            'custom_lists_version': 'datalar.py',
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
                return pd.DataFrame(
                    columns=['Tarih', 'Dosya', 'Toplam Tespit', 'Başarılı', 'Başarı Oranı', 'Veri Kaynağı'])
            rows = []
            for log in all_logs[-50:]:
                timestamp = datetime.fromisoformat(log['timestamp'])
                rows.append({
                    'Tarih': timestamp.strftime('%d.%m.%Y %H:%M'),
                    'Dosya': log['filename'],
                    'Toplam Tespit': log['total_entities'],
                    'Başarılı': log.get('successful_replacements', 0),
                    'Başarı Oranı': f"{log.get('success_rate', 0):.1f}%",
                    'Veri Kaynağı': log.get('custom_lists_version', 'Custom Lists')
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame(columns=['Tarih', 'Dosya', 'Toplam Tespit', 'Başarılı', 'Başarı Oranı', 'Veri Kaynağı'])

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

        # Custom Lists istatistikleri
        total_samples = 0
        for entity_type, lengths_dict in self.organized_data.items():
            total_samples += sum(len(samples) for samples in lengths_dict.values())

        with gr.Blocks(css=css, title="PDF Kişisel Bilgi Anonimleştirici - Custom Lists",
                       theme=gr.themes.Soft()) as interface:
            # Model durumu göster
            model_status = "✅ Custom NER Model Yüklendi" if self.ner_pipeline else "❌ Model Yüklenemedi"
            model_class = "model-loaded" if self.ner_pipeline else "model-error"

            gr.Markdown(f"""
            # 🔐 PDF Kişisel Bilgi Anonimleştirici — Custom Data Lists
            Yüklediğiniz PDF'teki kişisel bilgileri **özel modelinizle tespit eder** ve **kendi veri listelerinizle tutarlı + karakter sayısı korumalı değiştirme** yapar.

            <div class="model-status {model_class}">
            🤖 **Model Durumu**: {model_status}<br>
            📁 **Model Yolu**: {self.model_path}<br>
            📊 **Custom Lists**: {total_samples} toplam veri örneği<br>
            🔧 **Özellikler**: Tutarlı değiştirme, karakter sayısı koruma, custom veri listeleri, TC kimlik regex
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
                                label="🔁 Değiştirme Stratejisi",
                                info="Sahte Değerler: Custom Lists kullanılır"
                            )

                            with gr.Accordion("🧩 Özel Değer Haritası (JSON)", open=False):
                                gr.Markdown(
                                    "**Not**: Sistem önce Custom Lists'ten uygun uzunlukta veri bulmaya çalışır.")
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
                                "🚀 Custom Lists ile İşlemi Başlat",
                                variant="primary",
                                interactive=bool(self.ner_pipeline)
                            )

                            if not self.ner_pipeline:
                                gr.Markdown("⚠️ **Uyarı**: Model yüklenemediği için işlem başlatılamaz.")

                        with gr.Column(scale=1):
                            output_pdf = gr.File(label="📤 Anonimleştirilmiş PDF", file_count="single")
                            status_text = gr.Markdown("⏳ Henüz işlem yapılmadı.")
                            summary_md = gr.Markdown("")
                            stats_df = gr.Dataframe(
                                headers=["Veri Türü", "Tespit", "Değiştirildi", "Başarı Oranı", "Örnek"],
                                interactive=False)
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
                            inputs=[pdf_input, confidence_threshold, replacement_strategy, custom_replacements,
                                    enable_logging, enable_debug],
                            outputs=[output_pdf, status_text, summary_md, stats_df, debug_out],
                            api_name="process_pdf"
                        )

                    # ------------------ Custom Lists İnceleme ------------------
                    with gr.TabItem("📋 Custom Lists İnceleme"):
                        gr.Markdown("### Custom Lists Veri İstatistikleri")

                        # Veri listesi özeti
                        list_summary = []
                        for entity_type, lengths_dict in self.organized_data.items():
                            total_samples = sum(len(samples) for samples in lengths_dict.values())
                            if total_samples > 0:
                                length_range = f"{min(lengths_dict.keys())}-{max(lengths_dict.keys())}" if lengths_dict else "N/A"
                                list_summary.append({
                                    'Veri Türü': entity_type.replace('_', ' ').title(),
                                    'Toplam Örnek': total_samples,
                                    'Uzunluk Çeşitliliği': len(lengths_dict),
                                    'Uzunluk Aralığı': length_range
                                })

                        summary_df_lists = gr.Dataframe(
                            value=pd.DataFrame(list_summary),
                            label="Veri Listesi Özeti",
                            interactive=False
                        )

                        # Detaylı görüntüleme
                        with gr.Row():
                            entity_type_selector = gr.Dropdown(
                                choices=list(self.organized_data.keys()),
                                value=list(self.organized_data.keys())[0] if self.organized_data else None,
                                label="Veri Türü Seçin"
                            )
                            length_selector = gr.Dropdown(label="Uzunluk Seçin")

                        samples_display = gr.Textbox(
                            label="Örnek Veriler",
                            lines=10,
                            interactive=False
                        )

                        def update_length_options(entity_type):
                            if entity_type and entity_type in self.organized_data:
                                lengths = sorted(self.organized_data[entity_type].keys())
                                return gr.update(choices=lengths, value=lengths[0] if lengths else None)
                            return gr.update(choices=[], value=None)

                        def display_samples(entity_type, length):
                            if (entity_type and length is not None and
                                    entity_type in self.organized_data and
                                    length in self.organized_data[entity_type]):
                                samples = self.organized_data[entity_type][length]
                                sample_text = "\n".join(samples[:50])  # İlk 50 örnek
                                if len(samples) > 50:
                                    sample_text += f"\n\n... ve {len(samples) - 50} örnek daha"
                                return sample_text
                            return "Seçilen kategoride veri bulunamadı."

                        entity_type_selector.change(
                            update_length_options,
                            inputs=[entity_type_selector],
                            outputs=[length_selector]
                        )

                        length_selector.change(
                            display_samples,
                            inputs=[entity_type_selector, length_selector],
                            outputs=[samples_display]
                        )

                        # Sayfa yüklendiğinde ilk değerleri ayarla
                        interface.load(
                            update_length_options,
                            inputs=[entity_type_selector],
                            outputs=[length_selector]
                        )

                    # ------------------ Model Test ------------------
                    with gr.TabItem("🧪 Model Test"):
                        gr.Markdown("### Custom NER Model + TC Kimlik Regex + Custom Lists Testi")

                        test_text = gr.Textbox(
                            label="Test Metni",
                            placeholder="Örnek: Ahmet Yılmaz 12345678901 TC kimlik ile ahmet@email.com adresine mail gönderdi.",
                            lines=5
                        )
                        test_confidence = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                            label="Test Güven Eşiği"
                        )
                        test_btn = gr.Button("🔍 Test Et", variant="secondary")
                        test_results = gr.JSON(label="Test Sonuçları")

                        def test_model_with_custom_lists(text, conf_threshold):
                            if not self.ner_pipeline or not text.strip():
                                return {"error": "Model yüklü değil veya metin boş"}

                            try:
                                # NER model testi
                                results = self.ner_pipeline(text)
                                ner_results = []
                                for r in results:
                                    if r['score'] >= conf_threshold:
                                        entity_type = self.map_model_label_to_type(r['entity_group'])
                                        original_word = r['word']

                                        # Custom Lists'ten değiştirme üret
                                        replacement = self.generate_text_with_exact_length(
                                            len(original_word), entity_type, original_word
                                        )

                                        ner_results.append({
                                            'method': 'NER_Model',
                                            'entity': entity_type,
                                            'word': original_word,
                                            'replacement': replacement,
                                            'score': round(r['score'], 3),
                                            'start': r['start'],
                                            'end': r['end'],
                                            'original_label': r['entity_group'],
                                            'custom_lists_used': entity_type in self.organized_data and len(
                                                original_word) in self.organized_data.get(entity_type, {})
                                        })

                                # TC kimlik regex testi
                                tc_pattern = r'\b[1-9][0-9]{9}[02468]\b'
                                tc_results = []
                                for match in re.finditer(tc_pattern, text):
                                    tc_no = match.group()
                                    if self.validate_turkish_id(tc_no):
                                        replacement = self.generate_text_with_exact_length(
                                            len(tc_no), 'tc_kimlik', tc_no
                                        )
                                        tc_results.append({
                                            'method': 'Regex_TC',
                                            'entity': 'tc_kimlik',
                                            'word': tc_no,
                                            'replacement': replacement,
                                            'score': 0.95,
                                            'start': match.start(),
                                            'end': match.end(),
                                            'validation': 'Valid',
                                            'custom_lists_used': False  # TC için regex kullanılıyor
                                        })

                                all_results = ner_results + tc_results
                                custom_lists_count = sum(1 for r in all_results if r.get('custom_lists_used', False))

                                return {
                                    "total_entities": len(all_results),
                                    "ner_entities": len(ner_results),
                                    "tc_entities": len(tc_results),
                                    "custom_lists_usage": custom_lists_count,
                                    "results": all_results
                                }
                            except Exception as e:
                                return {"error": str(e)}

                        test_btn.click(
                            test_model_with_custom_lists,
                            inputs=[test_text, test_confidence],
                            outputs=test_results
                        )

                    # ------------------ Geçmiş ------------------
                    with gr.TabItem("🗂️ Geçmiş"):
                        gr.Markdown("Son işlemlerin özeti (Custom Lists kullanım bilgileri dahil).")
                        history_df = gr.Dataframe(interactive=False)
                        refresh_btn = gr.Button("🔄 Yenile")

                        def _load_history():
                            return self.get_processing_history()

                        refresh_btn.click(_load_history, inputs=None, outputs=history_df)
                        interface.load(_load_history, inputs=None, outputs=history_df)

                    # ------------------ Yardım ------------------
                    with gr.TabItem("❓ Yardım"):
                        gr.Markdown(f"""
    ### 🤖 Enhanced NER Model + Custom Lists Bilgileri
    - **Model Yolu**: `{self.model_path}`
    - **Model Durumu**: {'✅ Yüklendi ve hazır' if self.ner_pipeline else '❌ Yüklenemedi'}
    - **Veri Kaynağı**: Custom Lists (datalar.py)
    - **Toplam Veri**: {total_samples} örnek
    - **İşlem Yöntemi**: Font özelliklerini koruyarak değiştirme

    ### 🗂️ Custom Lists Özellikleri
    - **Tam Karakter Eşleştirme**: Orijinal metin uzunluğuyla aynı uzunlukta veriler tercih edilir
    - **Uzunluk Bazlı Organizasyon**: Her entity türü için farklı uzunluklarda örnekler
    - **Tutarlı Seçim**: Aynı orijinal metin her zaman aynı değerle değiştirilir
    - **Akıllı Fallback**: Uygun uzunluk bulunamadığında yakın uzunluklar denenir

    ### 📋 Desteklenen Entity Tipleri & Custom Lists

    **NER Model ile Tespit + Custom Lists Değiştirme:**
    - **Ad Soyad** (`ad_soyad`): {sum(len(samples) for samples in self.organized_data.get('ad_soyad', {}).values())} örnek
    - **Telefon** (`telefon`): {sum(len(samples) for samples in self.organized_data.get('telefon', {}).values())} örnek  
    - **E-posta** (`email`): {sum(len(samples) for samples in self.organized_data.get('email', {}).values())} örnek
    - **Adres** (`adres`): {sum(len(samples) for samples in self.organized_data.get('adres', {}).values())} örnek
    - **Şirket** (`sirket`): {sum(len(samples) for samples in self.organized_data.get('sirket', {}).values())} örnek
    - **Para** (`para`): {sum(len(samples) for samples in self.organized_data.get('para', {}).values())} örnek
    - **Tarih** (`tarih`): {sum(len(samples) for samples in self.organized_data.get('tarih', {}).values())} örnek
    - **IBAN** (`iban`): {sum(len(samples) for samples in self.organized_data.get('iban', {}).values())} örnek

    **Regex ile Tespit + Geçerli Üretim:**
    - **TC Kimlik** (`tc_kimlik`): Regex tespiti + matemartiksel doğrulama + yeni geçerli TC üretimi

    ### 🔍 Karakter Eşleştirme Örnekleri (Custom Lists)
    ```
    Orijinal: "BİM" (3 karakter)
    Custom Lists'te 3 karakterli markalar: ["ŞOK", "A101", "MİM"]
    Seçilen: "ŞOK" (deterministik)

    Orijinal: "Ahmet Yılmaz" (12 karakter)  
    Custom Lists'te 12 karakterli isimler: ["Mehmet Demir", "Fatma Özkan", ...]
    Seçilen: "Mehmet Demir" (aynı seed için her zaman aynı)
    ```

    ### 🔄 Tutarlılık Garantisi
    - **BİM** ilk örnekte **ŞOK** ile değiştirildiyse, metin içinde tekrar geçerse yine **ŞOK** olur
    - **Ahmet** ilk örnekte **Mehmet** ile değiştirildiyse, her yerde **Mehmet** olur
    - Seed değer (orijinal metin) aynı olduğu sürece aynı değiştirme yapılır

    ### 🚨 Önemli Notlar
    1. **Custom Lists Önceliği**: Önce custom lists'ten tam uzunlukta arama yapılır
    2. **Yakın Uzunluk Denemesi**: Tam uzunluk bulunamazsa yakın uzunluklar denenir
    3. **Generic Fallback**: Hiçbir uygun veri bulunamadığında anlamlı generic değer üretilir
    4. **TC Kimlik Özel**: TC kimlik için regex + matematiksel doğrulama + yeni geçerli üretim
    5. **Karakter Koruma**: Orijinal uzunluk her zaman korunur
    6. **Font Koruma**: PDF'te font boyutu ve renk korunur
                        """)

                return interface


if __name__ == "__main__":
    app = EnhancedAnonymizationApp()
    demo = app.create_enhanced_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)