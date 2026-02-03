import json
from pathlib import Path

def fix_notebook_val_path():
    nb_path = Path("colab_launcher.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    made_changes = False
    
    # We want to replace the val_images logic block
    # Logic to insert:
    # val_images = "/content/data/validation/images"
    # if not os.path.exists(val_images):
    #     val_images = "/content/custom_data/validation/images"
    # if not os.path.exists(val_images) and os.path.exists("/content/custom_data"):
    #     val_images = "/content/custom_data"
    
    target_string = 'val_images = "/content/custom_data/validation/images"'
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            skip_next = False
            for i, line in enumerate(cell["source"]):
                if skip_next:
                    skip_next = False
                    continue
                    
                if target_string in line:
                    print("Found target line. Injecting smarter path logic.")
                    # Inject improved logic
                    new_source.append('    # Check common locations for validation data (dataset.py moves them to data/)\n')
                    new_source.append('    search_paths = [\n')
                    new_source.append('        "/content/data/validation/images",\n')
                    new_source.append('        "/content/custom_data/validation/images",\n')
                    new_source.append('        "/content/custom_data"\n')
                    new_source.append('    ]\n')
                    new_source.append('    val_images = "/content/custom_data" # default fallback\n')
                    new_source.append('    for p in search_paths:\n')
                    new_source.append('        if os.path.exists(p) and any(os.scandir(p)):\n')
                    new_source.append('            val_images = p\n')
                    new_source.append('            break\n')
                    new_source.append('\n')
                    # Skipping the original logic lines to avoid duplication if running multiple times
                    # Original:
                    # val_images = "/content/custom_data/validation/images"
                    # if not os.path.exists(val_images) and os.path.exists("/content/custom_data"):
                    #      val_images = "/content/custom_data"
                    
                    # Hacky: I'll just comment out the old lines if I can find them strictly, 
                    # but simple replacement of the first line + eating the next 3 lines is risky if line counts change.
                    # Safest is to just REPLACE the target line with the new block and let the old 'if' checks run benignly 
                    # or swallow them.
                    
                    # Let's replace the whole specific block if it matches what I saw in file_view.
                    # Lines 154-157 in previous view.
                    
                    made_changes = True
                elif 'if not os.path.exists(val_images) and os.path.exists("/content/custom_data")' in line:
                    if made_changes: continue # clear this line
                elif '# Fallback to whatever remains in custom_data' in line:
                    if made_changes: continue
                elif 'val_images = "/content/custom_data"' in line and 'default fallback' not in line:
                    if made_changes: continue
                else:
                    new_source.append(line)
            
            cell["source"] = new_source

    if made_changes:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=4)
        print("Successfully updated notebook validation path logic.")
    else:
        print("No changes needed (or target line not found).")

if __name__ == "__main__":
    fix_notebook_val_path()
