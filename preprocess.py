import json
import re
from pathlib import Path

ABBREVIATIONS = {
    r'\btp\b': 'thành phố',
    r'\btp\.': 'thành phố',
    r'\bnn\b': 'nhân dân',
    r'\bnn\.': 'nhân dân',
    r'\bhdnd\b': 'hội đồng nhân dân',
    r'\bhdnd\.': 'hội đồng nhân dân',
    r'\bhdpp\b': 'hội đồng pháp phát',
    r'\bhdpp\.': 'hội đồng pháp phát',
    r'\bubnđ\b': 'ủy ban nhân dân',
    r'\bubnđ\.': 'ủy ban nhân dân',
    r'\bqh\b': 'quốc hội',
    r'\bqh\.': 'quốc hội',
    r'\bbvh\b': 'bộ văn hóa',
    r'\bbvh\.': 'bộ văn hóa',
    r'\bbgd\b': 'bộ giáo dục',
    r'\bbgd\.': 'bộ giáo dục',
    r'\bbca\b': 'bộ công an',
    r'\bbca\.': 'bộ công an',
    r'\bvn\b': 'việt nam',
    r'\bvn\.': 'việt nam',
    r'\bkm\b': 'kilômét',
    r'\bkm\.': 'kilômét',
    r'\bbl\b': 'bài liên',
    r'\bbl\.': 'bài liên',
    r'\bvd\b': 'ví dụ',
    r'\bvd\.': 'ví dụ',
    r'\bvb\b': 'văn bản',
    r'\bvb\.': 'văn bản',
    r'\bnd\b': 'nội dung',
    r'\bnd\.': 'nội dung',
    r'\btc\b': 'tài chính',
    r'\btc\.': 'tài chính',
    r'\bkv\b': 'khu vực',
    r'\bkv\.': 'khu vực',
    r'\bđvtnhh\b': 'đơn vị trách nhiệm hạn chế',
    r'\bđvtnhh\.': 'đơn vị trách nhiệm hạn chế',
    r'\bctcp\b': 'công ty cổ phần',
    r'\bctcp\.': 'công ty cổ phần',
    r'\bcnvclđ\b': 'công nhân viên chức lao động',
    r'\bcnvclđ\.': 'công nhân viên chức lao động',
}

def expand_abbreviations(text):
    """
    Chuyển đổi các viết tắt thành đầy đủ
    """
    for abbr, full_form in ABBREVIATIONS.items():
        text = re.sub(abbr, full_form, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    """
    Tiền xử lý văn bản: chuyển thành chữ thường và mở rộng viết tắt
    """
    text = text.lower()
    text = expand_abbreviations(text)
    return text

def load_and_preprocess_corpus(input_file, output_file):
    """
    Đọc file corpus.jsonl, tiền xử lý và loại bỏ trùng lặp
    """
    seen_ids = set()
    processed_samples = []
    
    print(f"Đang xử lý file {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    
                    if sample.get('_id') in seen_ids:
                        print(f"  Bỏ qua dòng {line_num}: _id '{sample.get('_id')}' đã tồn tại")
                        continue
                    
                    seen_ids.add(sample.get('_id'))
                    
                    if 'text' in sample and sample['text']:
                        sample['text'] = preprocess_text(sample['text'])
                    
                    if 'title' in sample and sample['title']:
                        sample['title'] = preprocess_text(sample['title'])
                    
                    processed_samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    print(f"  Lỗi JSON tại dòng {line_num}: {e}")
                    continue
        
        print(f"Đang ghi {len(processed_samples)} samples vào file {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ Hoàn tất! Đã xử lý {len(processed_samples)} samples")
        print(f"  - Loại bỏ {line_num - len(processed_samples)} samples trùng lặp")
        
    except FileNotFoundError:
        print(f"✗ Không tìm thấy file: {input_file}")
    except Exception as e:
        print(f"✗ Lỗi khi xử lý: {e}")

if __name__ == "__main__":
    input_path = "corpus.jsonl"
    output_path = "corpus_preprocessed.jsonl"
    
    load_and_preprocess_corpus(input_path, output_path)
