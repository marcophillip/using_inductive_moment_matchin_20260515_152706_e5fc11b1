import arxiv
import json
import os

def search_arxiv(query, max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in search.results():
        results.append({
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'year': result.published.year,
            'abstract': result.summary,
            'pdf_url': result.pdf_url,
            'entry_id': result.entry_id
        })
    return results

queries = [
    "inductive moment matching",
    "distributional reinforcement learning",
    "moment matching Q-learning",
    "uncertainty estimation Q-learning",
    "moment matching machine learning"
]

all_results = {}
for q in queries:
    print(f"Searching for: {q}")
    all_results[q] = search_arxiv(q)

with open('manual_arxiv_search.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("Search complete. Results saved to manual_arxiv_search.json")
