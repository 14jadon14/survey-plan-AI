import json
from pathlib import Path

def update_training_params():
    nb_path = Path("colab_launcher.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    made_changes = False
    
    # Target line in Cell 4 (Training)
    # !PYTHONPATH="{CODE_DIR}" python "{script_path}" --data_path /content/custom_data
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if "python \"{script_path}\" --data_path /content/custom_data" in line:
                    # We want to add --imgsz 1024 --epochs 50 explicitly
                    if "--imgsz" not in line:
                        print(f"Updating training command: {line.strip()}")
                        # Replace the line with the enhanced command
                        new_line = line.replace(
                            "--data_path /content/custom_data", 
                            "--data_path /content/custom_data --imgsz 1024 --epochs 50"
                        )
                        new_source.append(new_line)
                        made_changes = True
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell["source"] = new_source

    if made_changes:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated training parameters (imgsz=1024, epochs=50).")
    else:
        print("No changes needed (parameters might already be set).")

if __name__ == "__main__":
    update_training_params()
