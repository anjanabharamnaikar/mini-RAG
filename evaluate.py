import requests
import json
import pandas as pd

API_URL = "http://127.0.0.1:8000/ask"
QUESTIONS_FILE = "questions.json"

def query_api(question: str, mode: str) -> dict:
    """Sends a single question to the API and returns the top result."""
    payload = {
        "q": question,
        "k": 1,
        "mode": mode
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data['answer']:
            top_context = data['contexts'][0]
            return {
                "answer": data['answer'].replace('\n', ' ').strip()[:100] + "...",
                "source": top_context['title'],
                "score": f"{top_context['score']:.3f}"
            }
        else:
            return {
                "answer": f"**ABSTAINED**: {data['abstain_reason']}",
                "source": "N/A",
                "score": "N/A"
            }
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"API Error: {e}",
            "source": "N/A",
            "score": "N/A"
        }

def main():
    """Runs evaluation and prints a markdown table."""
    with open(QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    
    results = []
    print("Running evaluation...")
    for item in questions:
        q = item['question']
        print(f"Querying for: '{q}'")
        
        baseline_res = query_api(q, 'baseline')
        reranked_res = query_api(q, 'reranked')
        
        results.append({
            "Question": q,
            "Baseline Answer (Top Chunk)": baseline_res['answer'],
            "Baseline Source": baseline_res['source'],
            "Reranked Answer (Top Chunk)": reranked_res['answer'],
            "Reranked Source": reranked_res['source']
        })
        
    df = pd.DataFrame(results)
    
    print("\n\n--- Evaluation Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()