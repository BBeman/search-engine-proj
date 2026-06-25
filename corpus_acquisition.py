import requests
from pathlib import Path
import time


# Wikipedia rejects requests without a descriptive User-Agent (HTTP 403).
# No email required — a project identifier is enough.
HEADERS = {"User-Agent": "search-engine-proj/0.1 (learning project)"}


def get_titles_in_category(category: str):
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "categorymembers", "cmtitle": category,
                "cmtype": "page", "cmlimit": "30", "format": "json"},
        headers=HEADERS,
    )
    title_data = resp.json()   
    return title_data

def get_article_text(title):
    for attempt in range(3):
        resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "exlimit": "max", "format": "json"},
        headers=HEADERS,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            if resp.status_code !=200:
                print(f"attempt {attempt+1} for {title} got {resp.status_code}, waiting ...")
                time.sleep(60)
    return None

def main():
    title_list = []
    category = "Category:Machine learning"
    titles = get_titles_in_category(category)
    members = titles["query"]["categorymembers"]
    for member in members:
        title_list.append(member["title"])
    
    print(title_list)
    
    for t in title_list:
        try:
            articles = get_article_text(t)
            article_members = articles["query"]["pages"].values()
            for article_member in article_members:
                folder = Path("corpus")
                title = article_member["title"]
                safename = title.replace("/", "_") +".txt"
                path = folder / safename
                with open(path, "w", encoding="utf-8") as f:
                    f.write(title + "\n\n" + article_member.get("extract"))
                    time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print (f"skipped {t}: {e}")
            continue
            


if __name__ == "__main__":
    main()