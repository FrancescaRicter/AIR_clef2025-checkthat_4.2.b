# ir_baseline_final_timed.py

import time
import string, json, csv
import numpy as np
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1) NLTK setup
nltk.download('punkt',   quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet', quiet=True)

# 2) File paths
COLLECTION_PATH     = '../subtask4b_collection_data.pkl'
TRAIN_QUERY_PATH    = '../subtask4b_query_tweets_train.tsv'
DEV_QUERY_PATH      = '../subtask4b_query_tweets_dev.tsv'
OUT_TRAIN_PRED_PATH = 'predictions_train.tsv'
OUT_DEV_PRED_PATH   = '../predictions_dev.tsv'

# 3) Load data
df_col   = pd.read_pickle(COLLECTION_PATH)
df_train = pd.read_csv(TRAIN_QUERY_PATH, sep='\t', dtype={'post_id':str})
df_dev   = pd.read_csv(DEV_QUERY_PATH,   sep='\t', dtype={'post_id':str})

# 4) Preprocessing + unigram & bigram tokenizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize_and_ngrams(text: str):
    txt = (text or '').lower().translate(
        str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in nltk.word_tokenize(txt)
        if tok.isalpha() and tok not in stop_words
    ]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    return tokens + bigrams

# 5) Build the BM25 corpus: title×2 + abstract
titles    = df_col['title'].fillna('').tolist()
abstracts = df_col['abstract'].fillna('').tolist()
uids      = df_col['cord_uid'].tolist()

bm25_corpus = []
for t,a in zip(titles, abstracts):
    t_toks = tokenize_and_ngrams(t)
    a_toks = tokenize_and_ngrams(a)
    bm25_corpus.append(t_toks*2 + a_toks)  # title-boost ×2

# 6) Initialize BM25 with tuned params
bm25 = BM25Okapi(bm25_corpus, k1=1.0, b=0.9)

# 7) Retrieval function (top-5)
def retrieve_top5(text: str):
    q_toks = tokenize_and_ngrams(text)
    scores = bm25.get_scores(q_toks)
    top5   = np.argsort(scores)[::-1][:5]
    return [uids[i] for i in top5]

# 8) Run + evaluate function
def run_and_evaluate(df, split_name, out_path):
    # 8.1 Generate predictions and measure time
    print(f"▶ [{split_name}] Retrieving for {len(df)} queries…")
    start = time.perf_counter()
    df['preds'] = df['tweet_text'].map(lambda q: json.dumps(retrieve_top5(q)))
    elapsed = time.perf_counter() - start
    print(f"  • Retrieval & write took {elapsed:.2f}s")

    # 8.2 Write predictions
    df[['post_id','preds']].to_csv(
        out_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL
    )

    # 8.3 Compute MRR@5 & Top-1
    rr_sum, top1 = 0.0, 0
    for _, row in df.iterrows():
        true = row['cord_uid']
        pred = json.loads(row['preds'])
        if true in pred:
            pos = pred.index(true) + 1
            rr_sum += 1.0/pos
            top1  += (pos == 1)
    n = len(df)
    mrr5 = rr_sum / n
    acc1 = top1  / n
    print(f"▶ [{split_name}] MRR@5 = {mrr5:.3f}, Top-1 = {acc1:.3f}\n")
    return elapsed, mrr5, acc1

# 9) Execute for train & dev
train_time, train_mrr, train_acc1 = run_and_evaluate(
    df_train, 'TRAIN', OUT_TRAIN_PRED_PATH
)
dev_time, dev_mrr, dev_acc1       = run_and_evaluate(
    df_dev,   'DEV',   OUT_DEV_PRED_PATH
)

# 10) Summary
print("=== Summary ===")
print(f"TRAIN: time={train_time:.2f}s, MRR@5={train_mrr:.3f}, Top-1={train_acc1:.3f}")
print(f"  DEV: time={dev_time:.2f}s,   MRR@5={dev_mrr:.3f}, Top-1={dev_acc1:.3f}")



















