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

## Stage 2

Now with our index done, we move to querying it. The goal is that we take a query from the user. It could be one word or multiple, and we want to fetch the documents that contain that word or query. We also want operators in this implementation, so AND queries, OR queries, and NOT queries.

Example:

word1 and word2 gives us the documents shared by both words. This should be the same even if the user doesn't type "and", which is actually a bug we had earlier where only the first word was being queried.

word1 or word2 gives us both sets merged together, so all documents from either word without the duplicates.

word1 not word2 returns documents containing word1 but not word2.

Firstly I loaded the json dump we had in index.json, the one we created during indexing. Now we need to trace the structure of this json, because what we did was map an "index" key to our inverted index and a "doc_length" key to our doc lengths. But for querying we only need what is in the inverted index. The inverted index, if we recall, is a token mapped to the documents it is in, and in our case also the term frequency, which we don't need for querying (we will need it later for ranking).

That is what the get_docs function does. It checks if the query from the user is in the inverted index, then returns a sorted list of the documents matched to that query.

Now for the main algorithm, which is our two pointer merge. For the AND operator we do an intersection merge. We walk both sorted lists with two pointers and check if the current document in term1's list equals the current document in term2's list. If they are equal, that document is in both, so we add it and advance both pointers. If term1's current document is alphabetically less than term2's, we advance term1's pointer, otherwise we advance term2's.

Union does the same walk, but this time it appends term1's document when it is less than term2's, then advances term1, and similar for term2. Then we flush the leftovers, so we extend the result with the slice of whatever is left in each list after the loop ends.

For difference we only append term1's document when it is less than term2's, meaning term2 has moved past it and they are not equal. Then we flush the leftover term1 documents.

When I started this I built the functions using hardcoded input, just (a,c,d,e) and (a,b,c). Originally I accidentally forgot to sort the input. On a sorted list the merge gives you a,c, but not sorting it gives you just a, so you lose a whole document, which is not ideal. The two pointer merge only works on sorted lists, which is why get_docs sorts before anything reaches the merge.

Now the main part is our dispatcher, which splits the query into a list of words then checks for an operator. We then normalise the query so it matches the tokens in our inverted index. Originally I only let 2 sorted document lists go into our functions, but that doesn't allow for more than 2 word queries, so I had to chain it. I get the first term's documents and assign them to a result variable, then loop through the rest of the terms, calling our function with the running result and the next term's documents, reassigning result each pass. So on each chain run it rewrites result with the narrowed down answer. Lastly I added a rarest first optimisation to the AND queries, the rarest term being the one that appears in the fewest documents. We sort the terms by how many documents they are in, smallest first, then start the intersection with that one. This speeds up the query because an intersection can only shrink as we add more terms, so starting with the smallest list keeps the running result small and every later merge has less to iterate over. If we started with the most common term instead, the running result would be large and each merge would do more work. We don't do this for OR because a union only grows, so the order doesn't save anything, and we don't do it for NOT because NOT is binary, it only has two sides so there is no chain to reorder.

## Stage 3

Moving onto TF-IDF. This is the term frequency multiplied by the log of the total number of documents divided by the number of documents containing the term. It's used to see how important a word is to a document in our corpus. A common word is in many documents, so DF is large, so N/DF is near 1, so the log of that is near 0, meaning near zero weight. A rare word is in fewer documents, so N/DF is big, which means a big IDF. So IDF down-weights common words and boosts rare ones.

I already completed the boolean retrieval. What I needed now was to rank the query terms against the documents. We normalise the query again, and because during the creation of our inverted index in index.py we already stored term frequency and doc length, we have what we need. To score a document we loop through the index, fetch the term frequency for each document, the number of documents (N is just the count of doc_length entries), and the document frequency (how many documents contain the term), then put those into the tf-idf formula and rank our documents. One thing to note: we use the count of documents for IDF, but the actual stored document lengths go unused in tf-idf. That is exactly why tf-idf has no length normalisation, and it is the gap BM25 fills later.

For a multi word query, if a document is already in our results we add the new tf-idf onto its running score rather than overwriting it. Then using heapq we find the top k (k = 10) documents and return them. For this project I also added the boolean query output, a list of the documents the query maps to, as a filter before we rank. In our case this changed the 4th result: before, when I searched computer and ai, one document had so many instances of ai that it outranked documents containing computer, and filtering to the boolean candidates dropped it out.

I used sum, not cosine. Cosine adds length normalisation by treating documents as vectors and measuring the angle between them, but I'm deferring that to BM25 in the next stages, which handles it better.