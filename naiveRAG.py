import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

CORPUS_PATH  = "/kaggle/input/datasets/minhvb10/zalo-dataset/corpus_preprocessed.jsonl"
QUERIES_PATH = "/kaggle/input/datasets/minhvb10/zalo-dataset/queries_preprocessed.jsonl"
TEST_PATH    = "/kaggle/input/datasets/minhvb10/zalo-dataset/test.jsonl"

MODEL_NAME   = "finetuned"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_BATCH  = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_corpus_jsonl(path):
    doc_text_map = {}
    corpus_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            cid = str(d["_id"])
            text = f"{d.get('title', '')} {d.get('text', '')}".strip()
            doc_text_map[cid] = text
            corpus_ids.append(cid)
    return doc_text_map, corpus_ids

def load_queries_jsonl(path):
    queries_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            queries_dict[str(q["_id"])] = q.get("text", "").strip()
    return queries_dict

def load_qrels_jsonl(path):
    qrels_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            qid = str(r["query-id"])
            cid = str(r["corpus-id"])
            score = r.get("score", 0)
            if score > 0:
                if qid not in qrels_dict:
                    qrels_dict[qid] = set()
                qrels_dict[qid].add(cid)
    return qrels_dict

def embed(model, texts, batch_size=EMBED_BATCH):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

def build_faiss(embs):
    d = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index

def naive_rag_retrieve(index, corpus_ids, q_emb_single, top_k=30):
    D, I = index.search(np.array([q_emb_single]), top_k)
    return [corpus_ids[idx] for idx in I[0] if idx != -1]

def naive_rag_retrieve_with_rerank(index, corpus_ids, doc_text_map, q_emb_single, query_text, reranker, top_k_seeds=30, top_k_final=10):
    D, I = index.search(np.array([q_emb_single]), top_k_seeds)
    seed_ids = [corpus_ids[idx] for idx in I[0] if idx != -1]
    
    cross_inputs = []
    for cid in seed_ids:
        cross_inputs.append([query_text, doc_text_map[cid]])
        
    if not cross_inputs:
        return []
        
    scores = reranker.predict(cross_inputs, batch_size=256)
    
    reranked_results = sorted(zip(seed_ids, scores), key=lambda x: x[1], reverse=True)
    final_candidate_ids = [cid for cid, score in reranked_results]
    
    return final_candidate_ids[:top_k_final]

def run_comparison_evaluation(index, corpus_ids, doc_text_map, eval_queries, eval_q_embs, eval_qrels, reranker):
    k_list = [1, 5, 10]
    num_queries = len(eval_queries)
    
    metrics = {
        "Naive RAG (No Reranker)": {
            "hits": {k: 0 for k in k_list}, 
            "recalls": {k: 0.0 for k in k_list},
            "precisions": {k: 0.0 for k in k_list}
        },
        "Naive RAG (With Reranker)": {
            "hits": {k: 0 for k in k_list}, 
            "recalls": {k: 0.0 for k in k_list},
            "precisions": {k: 0.0 for k in k_list}
        }
    }
    
    for i in tqdm(range(num_queries), desc="Đang đánh giá Naive RAG"):
        qid, q_text = eval_queries[i]
        q_emb = eval_q_embs[i]
        gt_ids = eval_qrels[qid]
        
        preds_no_rerank = naive_rag_retrieve(index, corpus_ids, q_emb, top_k=30)
        preds_with_rerank = naive_rag_retrieve_with_rerank(index, corpus_ids, doc_text_map, q_emb, q_text, reranker, top_k_seeds=30, top_k_final=10)
        
        for method_name, preds in [("Naive RAG (No Reranker)", preds_no_rerank), ("Naive RAG (With Reranker)", preds_with_rerank)]:
            for k in k_list:
                preds_at_k = set(preds[:k])
                correct = preds_at_k.intersection(gt_ids)
                
                if len(correct) > 0: 
                    metrics[method_name]["hits"][k] += 1
                
                metrics[method_name]["recalls"][k] += len(correct) / len(gt_ids)
                
                metrics[method_name]["precisions"][k] += len(correct) / k

    print("\n" + "="*85)
    print("BẢNG SO SÁNH KẾT QUẢ: NAIVE RAG TRƯỚC VÀ SAU KHI DÙNG RERANKER")
    print("="*85)
    
    for method_name in metrics.keys():
        print(f"\n[{method_name}]")
        for k in k_list:
            avg_recall = metrics[method_name]["recalls"][k] / num_queries
            avg_precision = metrics[method_name]["precisions"][k] / num_queries
            avg_hit = metrics[method_name]["hits"][k] / num_queries
            
            if (avg_precision + avg_recall) > 0:
                avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            else:
                avg_f1 = 0.0
                
            print(f"  K={k:<2d} | Recall: {avg_recall:.4f} | Precision: {avg_precision:.4f} | F1: {avg_f1:.4f} | Hit: {avg_hit:.4f}") 

def main():
    print("1. Đang nạp Dữ liệu...")
    doc_text_map, corpus_ids = load_corpus_jsonl(CORPUS_PATH)
    corpus_texts = [doc_text_map[cid] for cid in corpus_ids]

    queries_dict = load_queries_jsonl(QUERIES_PATH)
    qrels_dict = load_qrels_jsonl(TEST_PATH)

    eval_queries_list = []
    for qid in qrels_dict.keys():
        if qid in queries_dict:
            eval_queries_list.append((qid, queries_dict[qid]))

    print(f"Đã nạp {len(corpus_ids)} tài liệu, sẵn sàng đánh giá trên {len(eval_queries_list)} câu hỏi có đáp án.")

    print("\n2. Đang nạp mô hình Embedding (Bi-Encoder)...")
    embed_model = SentenceTransformer(MODEL_NAME, device=device)
    if device == 'cuda': embed_model.half()

    print("Đang nhúng Corpus và Queries...")
    corpus_emb = embed(embed_model, corpus_texts)
    index = build_faiss(corpus_emb)

    eval_texts_only = [item[1] for item in eval_queries_list]
    eval_q_embs = embed(embed_model, eval_texts_only)

    print("\n3. Đang nạp mô hình Reranker (Cross-Encoder)...")
    reranker = CrossEncoder(RERANK_MODEL, max_length=1024, device=device)
    if device == 'cuda': reranker.model.half()

    print("\n4. Bắt đầu đánh giá so sánh...")
    run_comparison_evaluation(index, corpus_ids, doc_text_map, eval_queries_list, eval_q_embs, qrels_dict, reranker)

if __name__ == "__main__":
    main()