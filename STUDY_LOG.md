This is just a dump of what I do at each stage. I run through it as if I'm recalling my steps.

My aim for this project is a from scratch search engine that indexes a document collection and answers queries with ranked results.

Resources I used were a search engine implementation guide for understanding index creation and inverted indexes, and Intro to Information Retrieval from Stanford.

My motivation: if you check my last repo it was a langchain pipeline using hybrid RAG. I wanted to just do something from scratch, and it's more fun for conceptual hardening.

## Stage 0

I start off by initializing my project and creating my virtual environment and repo files. The main thing I need is corpus documents. I could just find a dump online of pure text files, or parse an XML file like the search engine article suggested. But for the sake of this project I felt like writing a corpus acquisition API call to Wikipedia, to pull the articles I want from a specific title derived from a category (in our case machine learning).

I use requests to make two calls to wiki: a title retrieval and an extract retrieval. I used the wiki API sandbox to check what I needed first, then wrote the query parameters and got their JSON into dict format. So that's our get titles.

I added this later, but I added retry logic to the articles function because wiki was rate limiting us with status 429 at every 10 fetches. So within 3 attempts (because we don't want to keep looping attempts) we check if it's a successful status code 200. If not, then we wait 1 minute.

I only want this acquisition file to run when I directly run it, just for scope, so I wrapped it in a name dunder if statement. Anyways, I wanted to start with category machine learning. I send that as my parameter to fetch titles, and I index into my dict to just take out the title and append it to my title list so I can see what it is fetching. Then I also loop through each title and get their articles one by one. Originally I joined them with "|", but the rate limiting issue made me do it individually. I indexed into pages, used the values function call to fetch all the values of pages, then looped through that and wrote the title and extract to the corpus path. Through debugging I added try logic, because I didn't initially know it was a rate limit. For our path I couldn't risk titles returning "/", so I replaced it with underscores. I encode to utf-8 so we don't have character conflicts. Then we just write to file.

### Normalization

Next stage was the normalization pipeline. So we loaded our data separately so we could map each file to a dict with a key value of file name. Separate function, so we don't just go through every txt file in the corpus and shove it into one list, which is what would happen had we done everything in normalise. Originally I had done that, and then realized my mistake when I ran the code.

The purpose of normalisation is that we clean the text or numerical scores and reduce data redundancy, so our queries match the relevant documents. We normalize the tokens so that they match even with small differences. So in this case we remove stop words, make everything lowercase (so for example apple and APPLE fall under the same token), then we tokenize by white space or non char/digit characters, then process the text using stemming to remove inflection endings, almost like clipping the end of the tokens. There are multiple stemming algorithms, but for this case we used the porter stemmer, which uses vowels and consonants and removes end characters based off the number of vowel consonant groups in that token. That's our normalise function. We then pass that into our results function, where we fetch each txt file and normalise the text within.

I mention token, so to be clear: a token in this case is the individual words grouped together for processing. A term is the normalized token stored in search, the same characters after normalisation. It could be an individual token or a sequence of tokens.

Another point is the stop word tradeoff. We do it to drop off words like the, is, and. This shrinks the index, but it also breaks queries that depend on them. Modern LLM RAG systems avoid removing stop words from the index, because of RAG's reliance on semantic dense vector embeddings.

Last point: I chose stemming over lemmatization because for lexical search I only need terms to match consistently, not to be real words. Stemming is faster, self contained, and collapses more word variants which helps recall, while lemmatization needs a dictionary and part of speech tagging for little retrieval benefit here as we are focusing on recall over precision. Also to note, modern transformer models use neither, they use subword tokenization because the model learns word structure itself. so "running" in stem would be "run" in lem would be "run" but in subword would be ["run", "ning"]. for exact keyword matching we would use stem or lem which is lossy but for lossless transformer use models learn from data so it doesn't need pre-collapse. 


## Stage 1

I started off by creating an index python file. We have already done our corpus ingestion, normalization of the corpus and now we have built the core data structure which is our inverted index. An inverted index is mapping terms to its posting list which is a list of document IDs the term appears in. Why is it called an inverted index? Because the original layout our dictionary is set from document to terms. What we wanted however, was normalized words mapped to documents. This makes our search much faster as we don't have to search each document one by one line by line to find terms we can simply query a term and match it to documents that it has. An explanation to a non technical would be you put a word in a search bar and it returns every page that contains that word. Ordering those pages by relevance comes later, that is the ranking stage, the inverted index itself just finds the matches. 

Now for implementation. I started off with a basic hardcoded dictionary so that I could test the code against a small dict rather than our entire normalized corpus. I started off creating a dict then looping through our keys, then their values then if the value wasn't in our dictionary creating a set. then to that set adding our key as a value. That way we match a token to the document. That was the base of the reverse index. 

However, we then wanted term frequency, so how often the token appeared in a doc, we wanted the length of each doc and we can't just add that to our set. what we then needed was dicts of dicts not sets we needed to get a count of our terms and add them to the document. Using counter we get a key value pairing of our term with its frequency looping through that we then used the same original layout but instead of a set created an empty dictionary if we don't have the token already in the dict and then we added key values token and key with term frequency. As for length of documents we just use len of each document's token list.

Next we needed to dump it into a json. so we created a combined dict of our inverted index and doc length, created and opened up a json file and wrote a json dump of our combined data. Last point another reason set wouldn't have worked over dict is it made the index json serializable as sets can't be dumped to json. a dict is structured like json.