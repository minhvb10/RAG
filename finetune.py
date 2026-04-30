import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "model_name": "bkai-foundation-models/vietnamese-bi-encoder",
    "output_dir": "./finetuned",
    "num_epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class LegalDataLoader:
    def __init__(self, train_path, queries_path, corpus_path):
        self.train_path = train_path
        self.queries_path = queries_path
        self.corpus_path = corpus_path
        self.queries = {}
        self.corpus = {}
        self.train_pairs = []
        
        self._load_data()
    
    def _load_data(self):
        logger.info("Loading queries...")
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.queries[item['_id']] = item['text']
        
        logger.info(f"Loaded {len(self.queries)} queries")
        
        logger.info("Loading corpus...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.corpus[item['_id']] = item['text']
        
        logger.info(f"Loaded {len(self.corpus)} documents")
        
        logger.info("Loading training pairs...")
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                query_id = item['query-id']
                corpus_id = item['corpus-id']
                score = item.get('score', 1)
                
                if query_id in self.queries and corpus_id in self.corpus:
                    self.train_pairs.append({
                        'query_id': query_id,
                        'corpus_id': corpus_id,
                        'score': score
                    })
        
        logger.info(f"Loaded {len(self.train_pairs)} training pairs")
    
    def get_examples(self) -> List[InputExample]:
        examples = []
        for pair in self.train_pairs:
            examples.append(InputExample(
                texts=[
                    self.queries[pair['query_id']],
                    self.corpus[pair['corpus_id']]
                ],
                label=pair['score']
            ))
        return examples


def main():
    logger.info("Starting BGE-M3 Fine-tuning (SentenceTransformers approach)")
    logger.info(f"Device: {CONFIG['device']}")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Loading model: {CONFIG['model_name']}")
    model = SentenceTransformer(CONFIG['model_name'], device=CONFIG['device'])
    
    logger.info("Loading data...")
    data_loader = LegalDataLoader(
        train_path="train.jsonl",
        queries_path="queries_preprocessed.jsonl",
        corpus_path="corpus_preprocessed.jsonl"
    )
    
    if len(data_loader.train_pairs) == 0:
        logger.error("No training pairs found. Check your file paths.")
        return
    
    examples = data_loader.get_examples()
    logger.info(f"Created {len(examples)} training examples")
    
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=CONFIG['batch_size'])
    
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    logger.info("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG['num_epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        optimizer_params={'lr': CONFIG['learning_rate']},
        show_progress_bar=True,
    )
    
    logger.info(f"Saving model to {output_dir}...")
    model.save(str(output_dir))
    
    logger.info("Fine-tuning completed!")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
