import json
from normalize import normalise
import math
import heapq

with open("index.json") as f:
    data = json.load(f)


def get_docs(query: str) -> list[str]:
    if query in data["index"]:
        return sorted(data["index"][query])
    return []


def intersect(a: list, b: list) -> list[str]:
    i = 0
    j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union(a: list, b: list) -> list[str]:
    i = 0
    j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def difference(a: list, b: list) -> list[str]:
    i = 0
    j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            j += 1
    result.extend(a[i:])
    return result


def dispatcher(query: str) -> list[str]:
    words = query.split()

    if "AND" in words or "and" in words:
        terms = normalise(query)
        sorted_terms = sorted(terms, key=lambda t: len(get_docs(t)))

        result = get_docs(sorted_terms[0])
        for term in sorted_terms[1:]:
            result = intersect(result, get_docs(term))
        return result

    elif "OR" in words or "or" in words:
        terms = normalise(query)
        result = get_docs(terms[0])
        for term in terms[1:]:
            result = union(result, get_docs(term))
        return result

    elif "NOT" in words or "not" in words:
        terms = normalise(query)
        left_docs = get_docs(terms[0])
        right_docs = get_docs(terms[1])
        return difference(left_docs, right_docs)

    else:
        terms = normalise(query)
        if not terms:
            return []
        sorted_terms = sorted(terms, key=lambda t: len(get_docs(t)))
        result = get_docs(sorted_terms[0])
        for term in sorted_terms[1:]:
            result = intersect(result, get_docs(term))
        return result


def ranking(query: str, k: int = 10) -> list[tuple[str, float]]:

    result = {}
    terms = normalise(query)
    candidates = dispatcher(query)

    for term in terms:
        if term in data["index"]:
            idf = math.log(len(data["doc_length"]) / len(data["index"][term]))

            for docs in data["index"][term]:
                if docs in candidates:
                    tf = len(data["index"][term][docs])
                    if docs not in result:
                        result[docs] = tf * idf
                    else:
                        result[docs] += tf * idf

    top = heapq.nlargest(k, result.items(), key=lambda pair: pair[1])
    return top


def bm25(query: str, k: int = 10) -> list[tuple[str, float]]:

    result = {}
    k1 = 1.5
    b = 0.75
    terms = normalise(query)
    candidates = dispatcher(query)
    avgdl = sum(data["doc_length"].values()) / len(data["doc_length"])

    for term in terms:
        if term in data["index"]:
            df = len(data["index"][term])
            n = len(data["doc_length"])
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for docs in data["index"][term]:
                if docs in candidates:
                    tf = len(data["index"][term][docs])
                    dl = data["doc_length"][docs]
                    if docs not in result:
                        result[docs] = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                    else:
                        result[docs] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))

    top = heapq.nlargest(k, result.items(), key=lambda pair: pair[1])
    return top


def phrase_search(phrase: str) -> list[str]:
    terms = normalise(phrase)
    if not terms:
        return []

    candidates = get_docs(terms[0])
    for term in terms[1:]:
        candidates = intersect(candidates, get_docs(term))

    result = []
    for doc in candidates:
        first_positions = data["index"][terms[0]][doc]
        for pos in first_positions:
            match = True
            for offset, term in enumerate(terms[1:], start=1):
                if (pos + offset) not in data["index"][term][doc]:
                    match = False
                    break
            if match:
                result.append(doc)
                break
    return result