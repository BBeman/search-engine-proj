from normalize import normalise_result
import json

def build_index():
    # inverted_index: term -> { document -> [positions of that term in that doc] }
    inverted_index = {}
    # nr: { document -> [normalised tokens, in order] }  (comes from normalize.py)
    nr = normalise_result()
    # doc_lengths: document -> how many tokens it has (its length)
    doc_lengths = {}

    for key in nr:                              # key = one document (filename)
        tokens = nr[key]                        # this document's list of tokens
        doc_lengths[key] = len(tokens)          # store this doc's length (token count)
        for pos, token in enumerate(tokens):    # pos = where the token sits, token = the term
            if token not in inverted_index:         # first time we've seen this term anywhere
                inverted_index[token] = {}          # start its {doc: positions} dict
            if key not in inverted_index[token]:    # first time this term shows up in this doc
                inverted_index[token][key] = []     # start its positions list for this doc
            inverted_index[token][key].append(pos)  # record where the term occurred

    # bundle both structures and save to disk, so querying can load them without rebuilding
    data = {"index": inverted_index, "doc_length": doc_lengths}

    with open("index.json", "w") as f:
        json.dump(data, f)

    return inverted_index, doc_lengths


if __name__ == "__main__":
    build_index()