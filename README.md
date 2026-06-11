# RAG

Hệ thống hỏi-đáp sử dụng Retrieval-Augmented Generation (RAG) để truy xuất thông tin từ corpus tài liệu với embedding models được fine-tune.

## Mô hình sử dụng

- **Embedding Model**: `bkai-foundation-models/vietnamese-bi-encoder` 
- **Reranking Model**: `BAAI/bge-reranker-v2-m3` 
- **Vector database**: FAISS 

## Yêu cầu

- Python >= 3.8
- CUDA (tuỳ chọn, để sử dụng GPU)

Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

## Kỹ thuật Fine-tuning

**Phương pháp**: `MultipleNegativesRankingLoss` (Contrastive Learning)
- Huấn luyện embedding model trên cặp (query, relevant document) với cấu hình:
  - Batch size: 32
  - Learning rate: 2e-5
  - Epochs: 3
  - Max length: 512 tokens

## Kết quả

- Sau khi áp dụng fine-tuning, hiệu quả truy hồi của pipeline Naive RAG có sử dụng reranker được cải thiện so với mô hình embedding gốc. Cụ thể, Recall@1 tăng từ 0.6345 lên 0.6485, Recall@5 tăng từ 0.8331 lên 0.8655, và Recall@10 tăng từ 0.8725 lên 0.8985.


