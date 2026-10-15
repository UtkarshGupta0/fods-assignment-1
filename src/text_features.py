import numpy as np
import matplotlib.pyplot as plt

from common import (SEED, cached_features, describe, log_result, make_split,
                    plot_embedding_2d)

NEWS_CATS = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
N_DOCS = 2000


def load_newsgroups(n=N_DOCS):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", categories=NEWS_CATS, random_state=SEED, shuffle=True,
                           remove=("headers", "footers", "quotes"))  # remove metadata shortcuts
    return np.array(d.data[:n], dtype=object), np.array(d.target[:n]), d.target_names


def load_imdb(n=N_DOCS):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=SEED).select(range(n))
    return np.array(ds["text"], dtype=object), np.array(ds["label"]), ["negative", "positive"]

def tfidf_by_hand_demo():

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are pets"]
    cv = CountVectorizer()
    tf = cv.fit_transform(docs).toarray().astype(float)  # (3 docs, V words) raw counts
    df = (tf > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    manual = tf * idf
    manual /= np.linalg.norm(manual, axis=1, keepdims=True)
    sk = TfidfVectorizer().fit_transform(docs).toarray()
    print("  TF-IDF by hand matches sklearn:", np.allclose(manual, sk))
    print("  vocabulary:", list(cv.get_feature_names_out()))
    print("  idf:", np.round(idf, 2), " <- 'the','sat','on' (in many docs) are down-weighted")


def contextual_vs_static_demo():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()
    sents = ["he sat on the bank of the river", "she deposited cash at the bank",
             "the bank approved my loan"]
    vecs = []
    for s in sents:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]  # (tokens, 768)
        pos = tok.convert_ids_to_tokens(enc["input_ids"][0]).index("bank")
        vecs.append(h[pos])
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"  cos(bank_river, bank_cash) = {cos(vecs[0], vecs[1]):.3f}")
    print(f"  cos(bank_river, bank_loan) = {cos(vecs[0], vecs[2]):.3f}")
    print(f"  cos(bank_cash,  bank_loan) = {cos(vecs[1], vecs[2]):.3f}   <- expected highest (both financial)")
    print("  A static embedding would give cosine = 1.000 for all pairs.")



def tfidf_features(texts, train_idx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,  # tf -> 1+log(tf): damps repeats
                          min_df=2, max_features=20000)
    vec.fit(texts[train_idx])
    return vec.transform(texts)


def glove_average_features(texts):
    import gensim.downloader as api
    from nltk.tokenize import RegexpTokenizer
    kv = api.load("glove-wiki-gigaword-100")  # ~128 MB download, 400k words
    tokenizer = RegexpTokenizer(r"[a-z']+")
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vs = [kv[w] for w in tokenizer.tokenize(t.lower()) if w in kv]
        if vs:
            out[i] = np.mean(vs, axis=0)
    return out


def sbert_features(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(list(texts), batch_size=64, show_progress_bar=True)


def run_experiment(tag, texts, y, class_names):
    print(f"\n-- {tag}: {len(texts)} documents, {len(class_names)} classes --")
    split = make_split(y)
    X_tfidf, t1 = cached_features(f"text_{tag}_tfidf", lambda: tfidf_features(texts, split[0]))
    X_glove, t2 = cached_features(f"text_{tag}_glove", lambda: glove_average_features(texts))
    X_sbert, t3 = cached_features(f"text_{tag}_sbert", lambda: sbert_features(texts))
    for n, a in [("TF-IDF (sparse)", X_tfidf), ("GloVe avg", X_glove), ("SBERT", X_sbert)]:
        describe(n, a)
    print(f"  TF-IDF sparsity: {1 - X_tfidf.nnz / np.prod(X_tfidf.shape):.4f} of entries are zero")
    print("Linear-probe benchmark:")
    for name, X, t, note in [("TF-IDF", X_tfidf, t1, "sparse, count-based"),
                             ("GloVe average", X_glove, t2, "static dense, 100-d"),
                             ("Sentence-BERT", X_sbert, t3, "contextual dense, 384-d")]:
        log_result(f"text_{tag}", name, X, y, split, t, note)
    plot_embedding_2d(X_sbert, y, class_names, f"{tag}: SBERT (PCA)", f"text_{tag}_sbert_pca.png")


def main():
    import nltk
    nltk.download("stopwords", quiet=True)
    print("== TEXT ==")
    print("\nDemo 1: TF-IDF computed by hand");
    tfidf_by_hand_demo()
    print("\nDemo 2: contextual embeddings");
    contextual_vs_static_demo()
    run_experiment("20newsgroups", *load_newsgroups())
    run_experiment("imdb", *load_imdb())


if __name__ == "__main__":
    main()
