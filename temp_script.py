import json
with open('src/output_MATHSENSEI/mixed/April_7.11_test_cache.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        pid = data.get('pid')
        error = data.get('wolfram_alpha_search:error')
        if error:
            print(f'PID {pid}: {error}')