import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "load_dataset(\"imagefolder\", data_dir=str(DATA_DIR), split=\"train\")" in src:
            new_src = [s.replace("load_dataset(\"imagefolder\", data_dir=str(DATA_DIR), split=\"train\")", "load_dataset(\"imagefolder\", data_dir=str(DATA_DIR), split=\"train\", download_mode=\"force_redownload\")") for s in cell['source']]
            cell['source'] = new_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook exactly match metric successfully added.")
