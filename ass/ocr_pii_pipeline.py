import os
import sys
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import re
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class ImagePreprocessor:

    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        return image.convert('L')

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def enhance_sharpness(image: Image.Image, factor: float = 2.0) -> Image.Image:
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def denoise(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def deskew(image: Image.Image, max_angle: float = 5.0) -> Image.Image:
        gray = np.array(image.convert('L'))

       
        angles = np.arange(-max_angle, max_angle + 0.5, 0.5)
        best_score = -1
        best_angle = 0

        for angle in angles:
            rotated = image.rotate(angle, fillcolor='white', expand=False)
            gray_rotated = np.array(rotated.convert('L'))

            h_projection = np.sum(gray_rotated < 128, axis=1)
            score = np.var(h_projection)

            if score > best_score:
                best_score = score
                best_angle = angle

        if abs(best_angle) > 0.1:
            print(f"   Detected skew angle: {best_angle:.2f}°")
            return image.rotate(best_angle, fillcolor='white', expand=False)
        return image

    def preprocess(self, image: Image.Image, config: Dict = None) -> Image.Image:
        if config is None:
            config = {
                'grayscale': True,
                'contrast': 1.8,
                'sharpness': 1.5,
                'denoise': True,
                'deskew': True
            }

        processed = image.copy()

     
        if config.get('deskew', False):
            processed = self.deskew(processed)

        
        if config.get('grayscale', True):
            processed = self.convert_to_grayscale(processed)

       
        if config.get('denoise', True):
            processed = self.denoise(processed)

       
        contrast_factor = config.get('contrast', 1.8)
        if contrast_factor > 1.0:
            processed = self.enhance_contrast(processed, contrast_factor)

       
        sharpness_factor = config.get('sharpness', 1.5)
        if sharpness_factor > 1.0:
            processed = self.enhance_sharpness(processed, sharpness_factor)

        return processed


class OCREngine:
   

    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.available = True
            print("  Tesseract OCR loaded successfully")
        except ImportError:
            print("  Warning: pytesseract not found. Install with: pip install pytesseract")
            print("  Also ensure Tesseract-OCR is installed on your system")
            self.available = False

    def extract_text(self, image: Image.Image, lang: str = 'eng', 
                     config: str = '') -> str:
        if not self.available:
            raise RuntimeError("Tesseract OCR not available. Please install pytesseract and Tesseract-OCR")

        try:
            if not config:
                config = '--psm 6 --oem 3'

            text = self.pytesseract.image_to_string(
                image, 
                lang=lang, 
                config=config
            )
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def extract_with_confidence(self, image: Image.Image) -> Dict:
        if not self.available:
            raise RuntimeError("Tesseract OCR not available")

        try:
            data = self.pytesseract.image_to_data(
                image, 
                output_type=self.pytesseract.Output.DICT
            )
            return data
        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': [], 'conf': []}
    
    def extract_high_quality_text(self, image, lang="eng", min_conf=50, config=""):
        if not self.available:
            raise RuntimeError("Tesseract OCR not available")

        try:
            if not config:
                config = "--psm 6 --oem 3"

            data = self.pytesseract.image_to_data(
                image,
                lang=lang,
                config=config,
                output_type=self.pytesseract.Output.DICT,
            )

            lines = {}
            n = len(data["text"])

            for i in range(n):
                word = data["text"][i]
                conf_str = data["conf"][i]

                try:
                    conf = float(conf_str)
                except:
                    conf = -1

                if not word.strip():
                    continue
                if conf < min_conf:
                    continue  

                key = (
                    data["block_num"][i],
                    data["par_num"][i],
                    data["line_num"][i],
                )
                lines.setdefault(key, []).append(word.strip())

            sorted_keys = sorted(lines.keys())
            joined_lines = [" ".join(lines[k]) for k in sorted_keys if lines[k]]

            return "\n".join(joined_lines).strip()

        except Exception as e:
            print("High-quality OCR error:", e)
            return ""



class TextCleaner:
    """Cleans and normalizes OCR-extracted text"""

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    @staticmethod
    def remove_special_characters(text: str, keep: str = '.,;:!?()-/') -> str:
        pattern = f'[^a-zA-Z0-9\s{re.escape(keep)}]'
        return re.sub(pattern, '', text)

    @staticmethod
    def fix_common_ocr_errors(text: str) -> str:
        replacements = {
            r'\b0([A-Z])': r'O\1',  
            r'([A-Z])0\b': r'\1O',  
            r'\b1([a-z])': r'I\1',  
            r'\|': 'I',  
            r'rn': 'm',  
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text

    def clean(self, text, aggressive=False):
        if not text:
            return ""

        cleaned = self.remove_extra_whitespace(text)

        if aggressive:
            cleaned = self.remove_special_characters(cleaned, keep=".,;:!?()-/%")
            cleaned = self.fix_common_ocr_errors(cleaned)

        return cleaned


class PIIDetector:

    def __init__(self):
        self.patterns = {
            'phone': [
                r'\b\d{10}\b',  
                r'\b\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}\b',  
                r'\+\d{1,3}[-\.\s]?\d{10}\b',  
                r'\b\d{5}[-\.\s]?\d{5}\b',  
            ],

            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],

            'medical_id': [
                r'\b(?:IPD|UHID|MRN|Patient\s+ID)\s*(?:No\.?|Number)?\s*:?\s*([A-Z0-9]+)\b',
                r'\b[A-Z]{2,4}\d{6,10}\b',  
            ],

            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            ],

            'name': [
                r'(?:Patient\s+Name|Name|Doctor)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                r'(?:Dr\.|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            ],

         
            'address': [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Nagar|Colony)\b',
                r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\s+\d{5,6}\b',
            ],

            'ssn': [
                r'\b\d{3}-\d{2}-\d{4}\b',  
                r'\b\d{4}\s\d{4}\s\d{4}\b',  
            ],

            
            'age': [
                r'\b(?:Age|AGE)\s*:?\s*(\d{1,3})\s*(?:Y|years?|yrs?)?\b',
            ],

            'bed_number': [
                r'\b(?:Bed|Ward)\s*(?:No\.?|Number)?\s*:?\s*([A-Z0-9-]+)\b',
            ],

            'department': [
                r'\b(?:Dept\.?|Department)\s*:?\s*([A-Z][A-Za-z\s&]+)\b',
            ]
        }

    def detect_all(self, text: str) -> Dict[str, List[Dict]]:
        """
        Detect all PII types in text

        Args:
            text: Input text

        Returns:
            Dictionary with PII type as key and list of matches
        """
        results = defaultdict(list)

        for pii_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    results[pii_type].append({
                        'value': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'pattern': pattern
                    })

        for pii_type in results:
            seen = set()
            unique = []
            for item in results[pii_type]:
                key = (item['value'], item['start'])
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            results[pii_type] = unique

        return dict(results)

    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """Get summary count of PII types detected"""
        all_pii = self.detect_all(text)
        summary = {pii_type: len(matches) for pii_type, matches in all_pii.items()}
        summary['total'] = sum(summary.values())
        return summary


class TextRedactor:

    def __init__(self, pii_detector: PIIDetector):
        self.detector = pii_detector

    def redact_text(self, text: str, pii_types: List[str] = None, 
                   redaction_char: str = 'X') -> str:
        all_pii = self.detector.detect_all(text)

        if pii_types:
            all_pii = {k: v for k, v in all_pii.items() if k in pii_types}

        all_matches = []
        for pii_type, matches in all_pii.items():
            for match in matches:
                match['type'] = pii_type
                all_matches.append(match)

        all_matches.sort(key=lambda x: x['start'], reverse=True)

        redacted = text
        for match in all_matches:
            start, end = match['start'], match['end']
            original_length = end - start
            replacement = redaction_char * original_length
            redacted = redacted[:start] + replacement + redacted[end:]

        return redacted


class OCRPIIPipeline:
  

    def __init__(self):
        
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.text_cleaner = TextCleaner()
        self.pii_detector = PIIDetector()
        self.text_redactor = TextRedactor(self.pii_detector)

    def process_image(self, image_path: str, 
                     preprocess_config: Dict = None) -> Dict:

        print(f"Processing: {image_path}")

        image = Image.open(image_path)
        print(f"    Image loaded: {image.size[0]}x{image.size[1]} pixels")

        # Preprocess
        preprocessed = self.preprocessor.preprocess(image, preprocess_config)
        print("    Preprocessing complete")

        raw_text = self.ocr_engine.extract_high_quality_text(
            preprocessed,
            lang="eng",
            min_conf=50   
        )
        print(f"    OCR complete (HQ): {len(raw_text)} characters kept")

        print(f"    OCR complete: {len(raw_text)} characters extracted")

        cleaned_text = self.text_cleaner.clean(raw_text)
        print("    Text cleaning complete")

        pii_detected = self.pii_detector.detect_all(cleaned_text)
        pii_summary = self.pii_detector.get_pii_summary(cleaned_text)
        print(f"    PII detection complete: {pii_summary['total']} instances found")

        redacted_text = self.text_redactor.redact_text(cleaned_text)
        print("    Text redaction complete")

        return {
            'image_path': image_path,
            'original_size': image.size,
            'extracted_text': cleaned_text,
            'pii_detected': pii_detected,
            'pii_summary': pii_summary,
            'redacted_text': redacted_text,
            'timestamp': datetime.now().isoformat()
        }

    def save_results(self, results: Dict, output_path: str):
        json_results = {
            'image_path': results['image_path'],
            'original_size': results['original_size'],
            'extracted_text': results['extracted_text'],
            'pii_summary': results['pii_summary'],
            'redacted_text': results['redacted_text'],
            'timestamp': results['timestamp'],
            'pii_details': {
                pii_type: [m['value'] for m in matches]
                for pii_type, matches in results['pii_detected'].items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("OCR + PII EXTRACTION PIPELINE")

    pipeline = OCRPIIPipeline()

    preprocess_config = {
        'grayscale': True,
        'contrast': 1.8,
        'sharpness': 1.5,
        'denoise': True,
        'deskew': True
    }

    image_files = ['samples/page_14.jpg', 'samples/page_30.jpg','samples/page_35.jpg']

    for image_file in image_files:
        if os.path.exists(image_file):
            print(f"\nProcessing {image_file}...")
            result = pipeline.process_image(image_file, preprocess_config)

            output_file = image_file.replace('.jpg', '_results.json')
            pipeline.save_results(result, output_file)
            print(f"    Results saved to {output_file}")
        else:
            print(f"    File not found: {image_file}")

    print("PROCESSING COMPLETE")
