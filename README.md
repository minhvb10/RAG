# RAG

Hệ thống hỏi-đáp sử dụng Retrieval-Augmented Generation (RAG) để truy xuất thông tin từ corpus tài liệu với embedding models được fine-tune.

## Mô hình sử dụng

- **Embedding Model**: `bkai-foundation-models/vietnamese-bi-encoder` (Vietnamese Bi-Encoder)
- **Reranking Model**: `BAAI/bge-reranker-v2-m3` (BGE Reranker v2-m3)
- **Retrieval**: FAISS (Facebook AI Similarity Search)

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

- Việc finetune embedding model đã cải thiện hiệu suất khi truy vấn, với việc không sử dụng reranker thì recall@1 tăng từ 0.42 lên 0.5 còn khi sử dụng reranker thì recall@1 tăng nhẹ từ 0.63 lên xấp xỉ 0.65


