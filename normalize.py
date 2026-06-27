from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
nltk.download("stopwords")  
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))  



def normalise(text):
    current_word = ""
    tokens = []
    for char in text:
        if char.isalnum():
            current_word += char.lower()
        else:
            if current_word != "":
                tokens.append(current_word)
                current_word = ""

    if current_word:
        tokens.append(current_word)

    tokens = [t for t in tokens if t not in STOPWORDS]          

    p_stemmer = PorterStemmer()
    stemmed_tokens = [p_stemmer.stem(token) for token in tokens]
    return stemmed_tokens
                    

def normalise_result():
    folder = Path("corpus")
    result = {}
    for file in folder.glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
        result[file.name] = normalise(text)
    return result

#print(normalise_result())