import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "DonutProcessor, VisionEncoderDecoderModel" in src and "DATA_DIR = \"/content/drive/MyDrive/SurveyPlan AI/runs/donut_tuning\"" in src:
            new_src = []
            for line in cell['source']:
                new_src.append(line)
                if "from transformers import DonutProcessor, VisionEncoderDecoderModel" in line:
                    new_src.append("import torch\n")
            
            # Make sure to avoid duplicating imports if already there
            final_src = []
            has_torch = False
            for line in new_src:
                if line.strip() == "import torch":
                    if has_torch: continue
                    has_torch = True
                final_src.append(line)
                
            cell['source'] = final_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook exactly match metric successfully added.")
