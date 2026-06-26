from normalize import normalise_result
from collections import Counter
import json

def build_index():
    inverted_index = {}
    nr = normalise_result()
    doc_lengths = {}

    for key in nr:
        counts = Counter(nr[key])
        doc_lengths[key] = len(nr[key])
        for token, tf in counts.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][key] = tf

    
    data = {"index": inverted_index, "doc_length": doc_lengths}

    with open("index.json", "w") as f:
        json.dump(data, f)

    return inverted_index, doc_lengths


if __name__ == "__main__":
    build_index()