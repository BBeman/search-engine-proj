import json 
from normalize import normalise
 

with open("index.json") as f:
    data = json.load(f)

def get_docs(query):
    if query in data["index"]:
        return sorted(data["index"][query])
    return []

            
def intersect(a,b):
    i = 0
    j = 0 
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i+=1
            j+=1
        elif a[i] < b[j]:
            i+=1
        else:
            j +=1
    return result
    

def union(a,b):
    i = 0
    j = 0 
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i+=1
            j+=1
        elif a[i] < b[j]:
            result.append(a[i])
            i+=1
        else:
            result.append(b[j])
            j +=1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

def difference(a,b):
    i = 0
    j = 0
    result =[]
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i+=1
            j+=1
        elif a[i] < b[j]:
            result.append(a[i])
            i+=1
        else:
            j +=1
    result.extend(a[i:])
    return result
    


def dispatcher(query):
    words = query.split()

    if "AND" in words or "and" in words:
        terms = normalise(query)
        sorted_terms = sorted(terms, key = lambda  t : len(get_docs(t)))

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
        sorted_terms = sorted(terms, key = lambda  t : len(get_docs(t)))
        result = get_docs(sorted_terms[0])
        for term in sorted_terms[1:]:
            result = intersect(result, get_docs(term))
        return result


print(dispatcher(input("search: ")))

