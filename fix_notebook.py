import json

file_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'processor.tokenizer.add_tokens(new_tokens)' in line:
                cell['source'][i] = line.replace('processor.tokenizer.add_tokens(new_tokens)', 'processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})')
                modified = True

if modified:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook successfully updated.")
else:
    print("Could not find the target string in the notebook.")
