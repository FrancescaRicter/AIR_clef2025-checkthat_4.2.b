# CheckThat! CLEF 2025 – Subtask 4b: Scientific Claim Source Retrieval

This repository contains our submission and experiments for the CLEF 2025 CheckThat! Lab, **Subtask 4b – Scientific Claim Source Retrieval**, which involves retrieving the correct scientific paper referenced implicitly in a tweet.

## Repository Structure

```
.
├── Neural_Representation_Learning_Approach/   # Sentence embedding + fine-tuning code
├── Traditional_IR/                            # BM25 and sparse retrieval scripts
├── getting_started_subtask4b.ipynb            # Official starter notebook
├── Neural_Ranking.ipynb                       # Main training & evaluation pipeline
├── predictions.tsv                            # Final predictions (full)
├── predictions_dev.tsv                        # Dev-set predictions
├── predictions_train.tsv                      # Train-set predictions
├── predictions.zip                            # Archived prediction files
├── submission.tsv                             # Final submission file (top 5 per tweet)
├── subtask4b_collection_data.pkl              # Paper metadata (CORD-19)
├── subtask4b_query_tweets_train.tsv           # Training tweets
├── subtask4b_query_tweets_dev.tsv             # Dev tweets
├── subtask4b_query_tweets_test.tsv            # Test tweets
├── README.md                                  # Project documentation
```

## Task Overview

Given a tweet that implicitly references a scientific publication, the goal is to retrieve the correct paper from a candidate pool derived from the CORD-19 dataset. This is framed as a document retrieval task evaluated using **MRR@5**.

More details about the challenge can be found here:  
[Official Task Page](https://checkthat.gitlab.io/clef2025/task4/)  
[Codalab Submission Platform](https://codalab.lisn.upsaclay.fr/competitions/22359)

## 🚀 Approaches

We implemented and compared the following pipelines:

### 1. Traditional Information Retrieval (BM25)
- Based on the `BM25Okapi` implementation from the `rank_bm25` library.
- Text preprocessing: tokenization, lemmatization, bigrams.
- Title field was emphasized (boosted) to improve retrieval performance.

### 2. Neural Representation Learning
- Used various pretrained sentence encoders (e.g., SciBERT, Specter2, Mixedbread).
- Best results were achieved using `mixedbread-ai/mxbai-embed-large-v1`.
- We explored contrastive fine-tuning using tweet-title and tweet-abstract pairs.

### 3. Neural Re-Ranking
- Used BM25 to retrieve top-30 documents.
- Applied pretrained cross-encoders (e.g., `ms-marco-MiniLM-L-12-v2`, `bge-reranker-large`) for re-ranking.
- Achieved best performance with `ms-marco-MiniLM-L-12-v2`.

## Performance Summary

| Approach                  | Dev MRR@5 |
|--------------------------|-----------|
| BM25 (Tuned)             | 0.638     |
| Neural Representation    | 0.607     |
| Neural Re-Ranking        | 0.626     |
| Challenge Submission     | 0.601     |

## Submission Format

The final `submission.tsv` file follows the required format:

```
post_id<TAB>[cord_uid1, cord_uid2, cord_uid3, cord_uid4, cord_uid5]
```

Each row contains a tweet ID and the top-5 predicted paper IDs sorted by relevance.

## Dependencies

To reproduce the results, install the following:

```bash
pip install rank_bm25 sentence-transformers transformers pandas nltk tqdm
```

For fine-tuning and re-ranking:
```bash
pip install accelerate datasets peft bitsandbytes
```

## Authors

This project was completed by students from TU Wien for the course **Advanced Information Retrieval 2025**:

- Trevor Calvin Baretto  
- Elias Hirsch  
- Francesca Ricter  
- Navya Velagaturi  
- Joseph Wagner  

## License

This repository is part of an academic project. Please refer to the [CheckThat! Challenge website](https://checkthat.gitlab.io/clef2025/task4/) for licensing information regarding datasets and evaluation code.
