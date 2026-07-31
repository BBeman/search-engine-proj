import query

# Relevance here is judged by hand, by me, by reading the documents. A document
# counts as relevant if it is about the query topic or gives it real discussion.
# A passing mention does not count, so "Outline of machine learning.txt" is never
# marked relevant, it is a page of links that names almost every term in the
# corpus without explaining any of them.
GROUND_TRUTH = {
    "algorithmic bias": {
        "Algorithmic bias.txt",
        "Artificial intelligence in hiring.txt",
        "Automated decision-making.txt",
        "80 Million Tiny Images.txt",
        "Machine learning.txt",
    },
    "neural network": {
        "A Logical Calculus of the Ideas Immanent in Nervous Activity.txt",
        "Attention (machine learning).txt",
        "Machine learning.txt",
        "Adversarial machine learning.txt",
        "Automated machine learning.txt",
    },
    "bayesian probability": {
        "Bayesian interpretation of kernel regularization.txt",
        "Bayesian learning mechanisms.txt",
        "Base rate.txt",
    },
    "anomaly detection": {
        "Anomaly detection.txt",
        "AIOps.txt",
    },
    "training data": {
        "Machine learning.txt",
        "80 Million Tiny Images.txt",
        "Adversarial machine learning.txt",
        "Active learning (machine learning).txt",
        "Automated machine learning.txt",
    },
}


def precision_at_k(results: list[tuple[str, float]], relevant: set[str], k: int) -> tuple[float, int, int]:
    top = [doc for doc, score in results[:k]]
    if not top:
        return 0.0, 0, 0
    hits = [doc for doc in top if doc in relevant]
    return len(hits) / len(top), len(hits), len(top)


def run_evaluation(k: int = 5) -> None:
    print(f"precision at {k}, {len(GROUND_TRUTH)} queries, relevance judged by hand")
    print("best is the ceiling, some queries have fewer than k relevant docs in the corpus")
    print()

    tfidf_scores = []
    bm25_scores = []
    ceilings = []

    for q in GROUND_TRUTH:
        relevant = GROUND_TRUTH[q]

        tfidf_results = query.ranking(q, k)
        bm25_results = query.bm25(q, k)

        tfidf_p, tfidf_hits, tfidf_n = precision_at_k(tfidf_results, relevant, k)
        bm25_p, bm25_hits, bm25_n = precision_at_k(bm25_results, relevant, k)

        # you cannot score higher than the number of relevant docs that exist
        ceiling = min(len(relevant), bm25_n) / bm25_n if bm25_n else 0.0

        tfidf_scores.append(tfidf_p)
        bm25_scores.append(bm25_p)
        ceilings.append(ceiling)

        print(f"query: {q}")
        print(f"  tf-idf  {tfidf_p:.2f}  ({tfidf_hits}/{tfidf_n} relevant)")
        print(f"  bm25    {bm25_p:.2f}  ({bm25_hits}/{bm25_n} relevant)")
        print(f"  best    {ceiling:.2f}  ({len(relevant)} relevant docs in the corpus)")

        for doc, score in bm25_results:
            mark = "+" if doc in relevant else "-"
            print(f"    {mark} {score:7.3f}  {doc}")
        print()

    tfidf_mean = sum(tfidf_scores) / len(tfidf_scores)
    bm25_mean = sum(bm25_scores) / len(bm25_scores)
    ceiling_mean = sum(ceilings) / len(ceilings)

    print(f"mean precision at {k}")
    print(f"  tf-idf  {tfidf_mean:.2f}")
    print(f"  bm25    {bm25_mean:.2f}")
    print(f"  best    {ceiling_mean:.2f}")


if __name__ == "__main__":
    run_evaluation()
