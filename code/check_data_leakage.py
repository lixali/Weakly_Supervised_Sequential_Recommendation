"""Check for data leakage (near-duplicate documents) between two text datasets.

Pipeline:
  1. Embed every document in both datasets with a multilingual sentence-transformer.
  2. Exact nearest-neighbor search (FAISS IndexFlatIP on L2-normalized vectors
     == exact cosine similarity) from every doc in the smaller set (clueweb)
     into the larger set (pixelrec).
  3. Verify the retrieved pairs lexically with word n-gram Jaccard/containment,
     because embedding similarity conflates "same topic" with "same text".
  4. Compare against a random-pair baseline so thresholds are calibrated,
     and dump the top pairs to CSV for manual reading.

Usage:
  python check_data_leakage.py \
      --query_csv  ~/HLLM_2_information/clueweb1000.csv \
      --corpus_csv ~/HLLM_2_information/Pixel200K_5_percent.csv \
      --out leakage_pairs.csv
"""

import argparse
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_text(df: pd.DataFrame, keys) -> pd.Series:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        raise ValueError(f"None of the text keys found in columns {list(df.columns)}")
    text = df[keys[0]].fillna("").astype(str)
    for k in keys[1:]:
        text = text + " " + df[k].fillna("").astype(str)
    return text.str.strip()


def word_ngrams(text: str, n: int) -> set:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def lexical_overlap(a: str, b: str, n: int):
    ga, gb = word_ngrams(a, n), word_ngrams(b, n)
    if not ga or not gb:
        return 0.0, 0.0
    inter = len(ga & gb)
    jaccard = inter / len(ga | gb)
    containment = inter / min(len(ga), len(gb))
    return jaccard, containment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_csv", required=True, help="smaller dataset (e.g. clueweb1000.csv)")
    ap.add_argument("--corpus_csv", required=True, help="larger dataset (e.g. Pixel200K_5_percent.csv)")
    ap.add_argument("--query_keys", nargs="+", default=["title", "tag", "description"])
    ap.add_argument("--corpus_keys", nargs="+", default=["title", "tag", "description"])
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ngram", type=int, default=5, help="word n-gram size for lexical check")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default="leakage_pairs.csv")
    args = ap.parse_args()

    query_df = pd.read_csv(args.query_csv)
    corpus_df = pd.read_csv(args.corpus_csv)
    query_text = build_text(query_df, args.query_keys)
    corpus_text = build_text(corpus_df, args.corpus_keys)
    print(f"query  docs: {len(query_text)}  ({args.query_csv})")
    print(f"corpus docs: {len(corpus_text)}  ({args.corpus_csv})")

    model = SentenceTransformer(args.model)
    print(f"encoding with {args.model} on {model.device} ...")
    q_emb = model.encode(query_text.tolist(), batch_size=args.batch_size,
                         normalize_embeddings=True, show_progress_bar=True)
    c_emb = model.encode(corpus_text.tolist(), batch_size=args.batch_size,
                         normalize_embeddings=True, show_progress_bar=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    c_emb = np.asarray(c_emb, dtype="float32")

    # Exact search: inner product on unit vectors == cosine similarity.
    index = faiss.IndexFlatIP(c_emb.shape[1])
    index.add(c_emb)
    sims, idxs = index.search(q_emb, args.topk)

    # Random-pair baseline to calibrate what "unrelated" looks like.
    rng = np.random.default_rng(0)
    qi = rng.integers(0, len(q_emb), 20000)
    ci = rng.integers(0, len(c_emb), 20000)
    baseline = (q_emb[qi] * c_emb[ci]).sum(axis=1)

    rows = []
    for i in range(len(q_emb)):
        for rank in range(args.topk):
            j = int(idxs[i, rank])
            jac, cont = lexical_overlap(query_text.iloc[i], corpus_text.iloc[j], args.ngram)
            rows.append({
                "query_id": query_df.get("item_id", pd.Series(range(len(query_df)))).iloc[i],
                "corpus_id": corpus_df.get("item_id", pd.Series(range(len(corpus_df)))).iloc[j],
                "rank": rank,
                "cosine": float(sims[i, rank]),
                f"jaccard_{args.ngram}gram": jac,
                f"containment_{args.ngram}gram": cont,
                "query_text": query_text.iloc[i][:300],
                "corpus_text": corpus_text.iloc[j][:300],
            })
    pairs = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    pairs.to_csv(args.out, index=False)

    max_sim = sims[:, 0]
    cont_col = pairs[f"containment_{args.ngram}gram"]
    print("\n================ SUMMARY ================")
    print(f"random-pair baseline cosine: mean={baseline.mean():.3f}  "
          f"p50={np.percentile(baseline, 50):.3f}  p99={np.percentile(baseline, 99):.3f}  "
          f"max={baseline.max():.3f}")
    print(f"nearest-neighbor cosine (per query doc): "
          f"p50={np.percentile(max_sim, 50):.3f}  p90={np.percentile(max_sim, 90):.3f}  "
          f"p99={np.percentile(max_sim, 99):.3f}  max={max_sim.max():.3f}")
    for t in (0.95, 0.90, 0.85, 0.80):
        print(f"query docs with a neighbor above cosine {t:.2f}: {(max_sim > t).sum()}")
    print(f"pairs with word-{args.ngram}-gram containment > 0.5 (near-verbatim): "
          f"{(cont_col > 0.5).sum()}")
    print(f"\ntop pairs written to {args.out} - read the top ~50 by hand.")
    print("Interpretation: cosine near baseline = unrelated; >0.85 with high "
          "n-gram containment = leakage; >0.85 with ~0 containment = same topic, "
          "different text (usually fine).")


if __name__ == "__main__":
    main()
