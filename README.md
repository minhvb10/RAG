# RAG

Hệ thống hỏi-đáp sử dụng Retrieval-Augmented Generation (RAG) để truy xuất thông tin từ corpus tài liệu và đánh giá chất lượng retrieval.

## Mô hình sử dụng

- **Embedding Model**: `bkai-foundation-models/vietnamese-bi-encoder` (Vietnamese Bi-Encoder)
- **Reranking Model**: `BAAI/bge-reranker-v2-m3` (BGE Reranker v2-m3)
- **Retrieval**: FAISS (Facebook AI Similarity Search)
- **LLM**: OpenAI GPT (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)

## Yêu cầu

- Python >= 3.8
- CUDA (tuỳ chọn, để sử dụng GPU)
- OpenAI API key

Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

## Quy trình sử dụng

### Bước 1: Fine-tune mô hình embedding
```bash
python finetune.py
```

### Bước 2: Xây dựng FAISS index
```bash
python naiveRAG.py
```

### Bước 3: Chạy ứng dụng web (Streamlit)
```bash
streamlit run app.py
```

## Cấu trúc dữ liệu

**Input Files:** 
- `data/corpus_preprocessed.jsonl` - Corpus tài liệu đã xử lý trước
- `data/train.jsonl` - Cặp (query, document) để training
- `data/queries_preprocessed.jsonl` - Queries đã xử lý trước
- `data/test.jsonl` - Ground truth để evaluation

