from normalize import normalise_result
import json

def build_index():
    inverted_index = {}
    nr = normalise_result()
    doc_lengths = {}

    for key in nr:
        tokens = nr[key]
        doc_lengths[key] = len(tokens)
        for pos, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if key not in inverted_index[token]:
                inverted_index[token][key] = []
            inverted_index[token][key].append(pos)

    
    data = {"index": inverted_index, "doc_length": doc_lengths}

    with open("index.json", "w") as f:
        json.dump(data, f)

    return inverted_index, doc_lengths


if __name__ == "__main__":
    build_index()