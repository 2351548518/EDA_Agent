import json
import os

BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")
input_path = os.path.join(BASE_OUTPUT_DIR, 'SFT_emb_reranker/data/emb_rank_train_raw.jsonl')
output_path = os.path.join(BASE_OUTPUT_DIR, 'SFT_emb_reranker/data/reranker_train.jsonl')

records = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        query = data['query']
        response = data['response']
        rejected_responses = data.get('rejected_response', [])

        # Build messages for positive sample (response is the correct answer)
        messages_pos = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: Given a search query, retrieve relevant passages that answer the query'\n<Query>: {query}\n<Document>: {response}"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\nyes"}
        ]

        # Build messages for negative samples (rejected_response is wrong answer)
        for rej in rejected_responses:
            messages_neg = [
                {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
                {"role": "user", "content": f"<Instruct>: Given a search query, retrieve relevant passages that answer the query'\n<Query>: {query}\n<Document>: {rej}"},
                {"role": "assistant", "content": "<think>\n\n</think>\n\nno"}
            ]
            records.append({"messages": messages_neg})

        records.append({"messages": messages_pos})

with open(output_path, 'w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"Converted {len(records)} records to {output_path}")
