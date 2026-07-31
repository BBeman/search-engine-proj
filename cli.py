import argparse
from pathlib import Path

INDEX_PATH = Path("index.json")


def run_index(args) -> None:
    from index import build_index

    inverted_index, doc_lengths = build_index()
    print(f"indexed {len(doc_lengths)} documents")
    print(f"{len(inverted_index)} unique terms")
    print(f"written to {INDEX_PATH}")


def run_search(args) -> None:
    if not INDEX_PATH.exists():
        print("no index.json found, run: python cli.py index")
        return

    # imported here and not at the top because query.py reads index.json as soon
    # as it is imported, so a top level import would break "cli.py index" on a
    # fresh clone where the index has not been built yet
    import query

    if args.phrase:
        docs = query.phrase_search(args.query)
        if not docs:
            print("no matches")
            return
        for rank, doc in enumerate(docs[: args.k], start=1):
            print(f"{rank:>2}. {doc}")
        return

    if args.method == "tfidf":
        results = query.ranking(args.query, args.k)
    else:
        results = query.bm25(args.query, args.k)

    if not results:
        print("no matches")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {score:8.3f}  {doc}")


def run_eval(args) -> None:
    if not INDEX_PATH.exists():
        print("no index.json found, run: python cli.py index")
        return

    from evaluate import run_evaluation

    run_evaluation(args.k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="a lexical search engine built from scratch over a wikipedia corpus"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build index.json from the corpus folder")
    index_parser.set_defaults(func=run_index)

    search_parser = subparsers.add_parser("search", help="run a query against the index")
    search_parser.add_argument("query", help="the search query")
    search_parser.add_argument(
        "--method",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="ranking method, default bm25",
    )
    search_parser.add_argument("-k", type=int, default=10, help="how many results to show, default 10")
    search_parser.add_argument(
        "--phrase",
        action="store_true",
        help="exact phrase search using the positional index",
    )
    search_parser.set_defaults(func=run_search)

    eval_parser = subparsers.add_parser("eval", help="run the precision evaluation")
    eval_parser.add_argument("-k", type=int, default=5, help="cutoff for precision at k, default 5")
    eval_parser.set_defaults(func=run_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
