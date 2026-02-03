import json
from pathlib import Path

def fix_notebook():
    nb_path = Path("colab_launcher.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    made_changes = False
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if "/detect/survey_plan_obb_run" in line:
                    print(f"Fixing line: {line.strip()}")
                    new_line = line.replace("/detect/", "/obb/")
                    new_source.append(new_line)
                    made_changes = True
                else:
                    new_source.append(line)
            cell["source"] = new_source
            
    if made_changes:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook paths to /obb/.")
    else:
        print("No changes needed (paths might already be correct).")

if __name__ == "__main__":
    fix_notebook()
