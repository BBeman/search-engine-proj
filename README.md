# Search Engine From Scratch

A lexical search engine written in plain Python, no search libraries. It pulls a corpus from Wikipedia, normalises the text, builds an inverted index, and answers queries with ranked results using tf-idf and BM25. It also handles boolean operators and exact phrase queries.

This is the retrieval layer that sits underneath modern RAG, built by hand so I actually understand what a vector database is replacing.

## Why I built it

My last project was a LangChain pipeline doing hybrid RAG. Hybrid means dense vector search plus a keyword search like BM25, and I realised I was calling BM25 without knowing what it did. So I went and built the keyword half from scratch.

Everything here is standard library plus NLTK for stopwords and the Porter stemmer. No Elasticsearch, no Whoosh, no scikit-learn. The two pointer merges, the scoring functions and the positional index are all written out.

## What it does

- Corpus acquisition from the Wikipedia API, with retry logic for rate limits
- Text normalisation: tokenising, lowercasing, stopword removal, Porter stemming
- An inverted index mapping each term to the documents it appears in and its positions in them
- Boolean queries with AND, OR and NOT, using two pointer merges on sorted posting lists
- Rarest term first optimisation on AND queries
- tf-idf ranking
- BM25 ranking with term frequency saturation and document length normalisation
- Exact phrase search using the positional index
- A small hand judged evaluation comparing tf-idf against BM25

## Architecture

The pipeline runs in four stages, each one is its own file.

```
Wikipedia API
     |
     |  corpus_acquisition.py     fetch articles by category
     v
corpus/*.txt                      30 plain text articles
     |
     |  normalize.py              tokenise, lowercase, drop stopwords, stem
     v
tokens per document
     |
     |  index.py                  invert it, record positions and doc lengths
     v
index.json                        {"index": {...}, "doc_length": {...}}
     |
     |  query.py                  boolean filter, then score and rank
     v
ranked results
     |
     |  cli.py                    the interface
     v
you
```

**normalize.py** turns raw text into terms. It walks the string character by character building tokens out of alphanumeric runs, lowercases them, drops English stopwords, then runs the Porter stemmer so that "learning", "learns" and "learned" all collapse to the same term. Both indexing and querying go through this same function, which matters, because the query has to be normalised exactly the same way as the documents were or nothing will match.

**index.py** builds the inverted index. A normal layout maps a document to the terms inside it. Inverting that gives you a term mapped to the documents it appears in, which is the whole trick, because now a lookup is one dictionary access instead of scanning every document line by line. Each posting also stores the positions of the term inside that document, so term frequency is just the length of the positions list. Document lengths are stored alongside it, and BM25 needs those later. All of it goes to `index.json`.

**query.py** does retrieval in two steps. First the boolean layer narrows the corpus to candidate documents, walking sorted posting lists with two pointer merges for intersection, union and difference. Then the ranking layer scores only those candidates with tf-idf or BM25 and returns the top k with a heap.

**cli.py** wires it together into commands.

## Setup

Needs Python 3.13 or newer. The project uses uv.

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install nltk requests
```

NLTK downloads the stopword list on first run, so the first command will print a download line.

## Usage

The corpus is already committed, so you only need to build the index.

```bash
python cli.py index
```

```
indexed 30 documents
4910 unique terms
written to index.json
```

Then search. BM25 is the default.

```bash
python cli.py search "algorithmic bias" -k 5
```

```
 1.     4.493  Algorithmic bias.txt
 2.     3.838  Automated decision-making.txt
 3.     3.762  Artificial intelligence in hiring.txt
 4.     3.287  Machine learning.txt
 5.     3.168  Outline of machine learning.txt
```

Switch the scorer with `--method`:

```bash
python cli.py search "algorithmic bias" --method tfidf -k 5
```

Exact phrase search, which uses the stored positions rather than just checking both words are present somewhere:

```bash
python cli.py search "training data" --phrase
```

Run the evaluation:

```bash
python cli.py eval
```

If you want to rebuild the corpus from Wikipedia instead of using the committed one:

```bash
python corpus_acquisition.py
```

That pulls 30 articles from Category:Machine learning. It sleeps on rate limits so it takes a few minutes.

## tf-idf vs BM25

This is the most interesting part of the project, so here it is side by side. Same query, same index, same boolean candidates, only the scoring function changes.

Query: `algorithmic bias`

| rank | tf-idf | len | BM25 | len |
|------|--------|-----|------|-----|
| 1 | Algorithmic bias.txt | 6720 | Algorithmic bias.txt | 6720 |
| 2 | Machine learning.txt | 5471 | Automated decision-making.txt | 1889 |
| 3 | Outline of machine learning.txt | 3041 | Artificial intelligence in hiring.txt | 864 |
| 4 | Automated decision-making.txt | 1889 | Machine learning.txt | 5471 |
| 5 | Adversarial machine learning.txt | 3664 | Outline of machine learning.txt | 3041 |

Average document length in this corpus is about 1357 tokens.

Look at the tf-idf column. Ranks 2 and 3 are the two longest general articles in the corpus. Neither is really about algorithmic bias, they are broad survey pages that mention it in passing. They rank that high mostly because they are big, and a big document has more room to contain any given term. tf-idf multiplies raw term frequency by idf and never asks how long the document was, so length leaks straight into the score.

BM25 fixes that with two changes. First it divides term frequency by a length factor, so a hit in a 6720 token article counts for less than a hit in an 864 token article. Second it saturates term frequency, so the 50th occurrence of a term adds far less than the 5th did. The `k1` parameter controls how fast saturation kicks in and `b` controls how strongly length is normalised, and I used the usual defaults of 1.5 and 0.75.

The result is that "Artificial intelligence in hiring.txt", which is 864 tokens and genuinely about biased hiring algorithms, climbs from outside the top 5 up to rank 3, and the two long survey pages get pushed down.

The raw scores are worth a note too. tf-idf gives the top document 246.7 and BM25 gives it 4.49. That is not BM25 being worse, the numbers just are not comparable, because tf-idf sums unbounded term frequencies while BM25 sums saturated ones. Only the ordering within a single scorer means anything.

## Evaluation

Five queries, with relevance judged by hand by reading the documents. A document counts as relevant if it is about the topic or gives it real discussion. A passing mention does not count, so "Outline of machine learning.txt" is never marked relevant, it is a page of links that names nearly every term in the corpus without explaining any of them.

Measured with precision at 5, so out of the top 5 results, how many were actually relevant.

| query | tf-idf | BM25 | best possible |
|-------|--------|------|---------------|
| algorithmic bias | 0.60 | **0.80** | 1.00 |
| neural network | 0.60 | **0.80** | 1.00 |
| bayesian probability | 0.40 | 0.40 | 0.60 |
| anomaly detection | 0.40 | 0.40 | 0.40 |
| training data | 0.40 | 0.40 | 1.00 |
| **mean** | **0.48** | **0.56** | **0.80** |

The best possible column matters. This corpus only has 30 documents, so some queries simply do not have 5 relevant documents in it. Only 2 documents in the whole corpus are about anomaly detection, which caps precision at 5 at 0.40, so BM25 scoring 0.40 there is actually a perfect result and not a bad one.

BM25 beats tf-idf on two queries and ties on the other three, so it never loses. Mean goes from 0.48 to 0.56 against a ceiling of 0.80. That is a real improvement but a modest one, which is about what you should expect from 5 queries on 30 documents. This is a sanity check that the ranking works, not a benchmark.

Where it still gets things wrong is more interesting than the score:

- **Polysemy.** For "neural network", the article on anomaly detection scores well because it is full of the word "network", except it means computer networks, sensor networks and network intrusion, not neural ones. Lexical search matches characters and has no idea the same string means two different things.
- **Long survey pages.** BM25 knocks these down but does not eliminate them. "Outline of machine learning.txt" is a list of links and still reaches the top 5 on most queries, because it genuinely does contain every term.
- **Stopwords and phrases.** Stopwords are stripped before positions are recorded, so positions are consecutive over surviving tokens and not over the original text. A phrase search for something like "the cat" cannot work, since "the" is not in the index at all.

## Files

| file | what it does |
|------|--------------|
| `corpus_acquisition.py` | pulls articles from the Wikipedia API into `corpus/` |
| `normalize.py` | tokenising, lowercasing, stopword removal, Porter stemming |
| `index.py` | builds the positional inverted index into `index.json` |
| `query.py` | boolean merges, tf-idf, BM25, phrase search |
| `cli.py` | the command line interface |
| `evaluate.py` | the hand judged precision evaluation |
| `STUDY_LOG.md` | my notes at every stage, written as I went |

`index.json` is gitignored since it is generated, so build it before searching.

## What I would add next

Things I deliberately stopped short of, since the goal was understanding the core rather than shipping a product:

- Compressing the index. Postings are stored as raw JSON integers right now, and real engines use delta encoding plus variable byte codes to shrink that a lot.
- Skip pointers on posting lists, so intersection can jump ahead instead of stepping one document at a time.
- Proximity scoring, where terms near each other score higher, rather than phrase matching being all or nothing.
- Query expansion with synonyms, which is one of the ways you deal with the polysemy problem above.
- A dense vector index next to this one, then combining the two with reciprocal rank fusion, which is what hybrid RAG actually is and where this started.
