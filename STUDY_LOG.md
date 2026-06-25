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
