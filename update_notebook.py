import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # 1. Update Load Dataset cell
        if "load_dataset(\"imagefolder\"" in src and "Pre-process metadata.jsonl" not in src:
            new_src = [
                "from datasets import load_dataset\n",
                "from pathlib import Path\n",
                "import json\n",
                "import os\n",
                "\n",
                "# Path where the metadata.jsonl and crops are stored\n",
                "DATA_DIR = Path('/content/drive/MyDrive/SurveyPlan AI/runs/donut_tuning')\n",
                "os.makedirs(DATA_DIR, exist_ok=True)\n",
                "\n",
                "try:\n",
                "    # Pre-process metadata.jsonl to remove missing images before loading\n",
                "    metadata_path = DATA_DIR / 'metadata.jsonl'\n",
                "    if metadata_path.exists():\n",
                "        valid_entries = []\n",
                "        missing_count = 0\n",
                "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
                "            for line in f:\n",
                "                entry = json.loads(line)\n",
                "                img_path = DATA_DIR / entry['file_name']\n",
                "                if img_path.exists():\n",
                "                    valid_entries.append(line)\n",
                "                else:\n",
                "                    missing_count += 1\n",
                "                    print(f\"Skipping missing file: {entry['file_name']}\")\n",
                "        \n",
                "        if missing_count > 0:\n",
                "            print(f\"Removed {missing_count} missing entries from metadata.jsonl\")\n",
                "            with open(metadata_path, 'w', encoding='utf-8') as f:\n",
                "                f.writelines(valid_entries)\n",
                "\n",
                "    # Load dataset using HuggingFace's imagefolder script\n",
                "    dataset = load_dataset(\"imagefolder\", data_dir=str(DATA_DIR), split=\"train\")\n",
                "    dataset = dataset.train_test_split(test_size=0.1)\n",
                "    train_dataset = dataset['train']\n",
                "    val_dataset = dataset['test']\n",
                "    print(f\"SUCCESS: Loaded {len(train_dataset)} train and {len(val_dataset)} validation samples.\")\n",
                "except Exception as e:\n",
                "    print(f\"ERROR LOADING DATASET: {e}\")"
            ]
            cell['source'] = new_src
            
        # 2. Update Start Training cell
        if "trainer.fit(pl_module," in src and "model.train()" not in src:
            new_src = []
            for line in cell['source']:
                if line.startswith("trainer.fit("):
                    new_src.extend([
                        "# Force model back into train mode (in case inference was run earlier)\n",
                        "model.train()\n",
                        "for param in model.parameters():\n",
                        "    param.requires_grad = True\n",
                        "\n",
                        "print(\"Model is now in TRAIN mode and weights are unfrozen.\")\n",
                        line
                    ])
                else:
                    new_src.append(line)
            cell['source'] = new_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook successfully updated.")
