## 1. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install Tesseract (macOS)
brew install tesseract

# Install Tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 2. Run the Pipeline

```bash
# Process sample documents
python ocr_pii_pipeline.py
```

## 3. Check Results

```bash
# View generated JSON files
cat page_30_results.json
cat page_35_results.json
cat page_14_results.json
```

## 4. Customize for Your Documents

Edit `ocr_pii_pipeline.py`:
- Update `image_files` list with your filenames (line ~450)
- Adjust preprocessing config if needed (line ~440)
- Modify PII patterns for your use case (class PIIDetector)

## File Structure

```
project/
├── ocr_pii_pipeline.py       # Main pipeline code
├── requirements.txt           # Dependencies
├── RESULTS_SUMMARY.md         # Detailed results
├── README.md                  # This file
├── page_30.jpg               # Sample input
├── page_35.jpg               # Sample input
├── page_14.jpg               # Sample input
├── page_30_results.json      # Output
├── page_35_results.json      # Output
└── page_14_results.json      # Output
```

## Support

- Python version: 3.8+
- Image formats: JPEG, PNG
- Document types: Medical records, forms, handwritten notes
