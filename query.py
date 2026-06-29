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


def ranking(query: str, k: int = 10) -> list[str]:
    result = {}
    terms = normalise(query)
    candidates = dispatcher(query)
    for term in terms:
        if term in data["index"]:
            idf = math.log(len(data["doc_length"]) / len(data["index"][term]))
            for docs in data["index"][term]:
                if docs in candidates:
                    tf = data["index"][term][docs]
                    if docs not in result:
                        result[docs] = tf * idf
                    else:
                        result[docs] += tf * idf

    top = heapq.nlargest(k, result, key=lambda d: result[d])
    return top


if __name__ == "__main__":
    # print(dispatcher(input("search: ")))
    print(ranking(input("search: ")))
