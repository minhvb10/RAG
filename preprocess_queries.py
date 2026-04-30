import json
import re
from pathlib import Path

# Từ điển chuyển đổi viết tắt thành đầy đủ (giống như trong preprocess.py)
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
    # Chuyển sang chữ cái thường
    text = text.lower()
    # Mở rộng các viết tắt
    text = expand_abbreviations(text)
    return text

def preprocess_queries(input_file, output_file):
    """
    Đọc queries.jsonl, tiền xử lý text
    """
    processed_count = 0
    
    print(f"Đang xử lý file {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                try:
                    query = json.loads(line.strip())
                    
                    # Tiền xử lý text
                    if 'text' in query and query['text']:
                        query['text'] = preprocess_text(query['text'])
                    
                    # Ghi ra file output
                    f_out.write(json.dumps(query, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"  Lỗi JSON tại dòng {line_num}: {e}")
                    continue
        
        print(f"✓ Hoàn tất! Đã xử lý {processed_count} queries")
        print(f"  - Output: {output_file}")
        
    except FileNotFoundError:
        print(f"✗ Không tìm thấy file: {input_file}")
    except Exception as e:
        print(f"✗ Lỗi khi xử lý: {e}")

if __name__ == "__main__":
    # Đường dẫn input và output
    input_path = Path(__file__).parent / "queries.jsonl"
    output_path = Path(__file__).parent / "queries_preprocessed.jsonl"
    
    # Xử lý file
    preprocess_queries(str(input_path), str(output_path))
