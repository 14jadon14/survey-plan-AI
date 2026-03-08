import json
import os

def update_notebook(filepath):
    print(f"Updating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Look for a cell that installs requirements or just add a new code cell at the top
    # Let's search for pip install inside cells
    inserted = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if '!pip install' in source:
                if 'deskew' not in source:
                    # Append it to the first found pip install cell
                    cell['source'].append('\n!pip install deskew')
                    inserted = True
                    break
                else:
                    print("deskew already installed in cell")
                    inserted = True
                    break
                    
    if not inserted:
        # Just prepend a new cell if we didn't find a pip install cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install deskew\n"
            ]
        }
        nb['cells'].insert(0, new_cell)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Updated {filepath}")

base_dir = r"c:\Users\Jadon\Desktop\SurveyPlan AI"
notebooks = [
    os.path.join(base_dir, "document_understanding", "colab_document_parser.ipynb"),
    os.path.join(base_dir, "yolo_training", "colab_launcher.ipynb")
]

for nb_path in notebooks:
    if os.path.exists(nb_path):
        update_notebook(nb_path)
    else:
        print(f"Not found: {nb_path}")
