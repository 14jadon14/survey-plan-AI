import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Append Active Learning cells
markdown_cell = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "## 10. Active Learning Auto-Labeler\n",
    "Once you have used your YOLO pipeline to save *new* blank crops to your Drive (where `ground_truth` is `{\"gt_parse\": {\"text\": \"\"}}`), you can run this cell! It will load your newly fine-tuned Donut model, predict the correct JSON tags for the new images, and overwrite the blank rows in your `metadata.jsonl` with the AI's best guess. This is the **Data Flywheel**!"
  ]
}

code_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "import os\n",
    "import json\n",
    "from PIL import Image\n",
    "from tqdm import tqdm\n",
    "from transformers import DonutProcessor, VisionEncoderDecoderModel\n",
    "\n",
    "# Ensure model is loaded\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "model.to(device)\n",
    "model.eval()\n",
    "\n",
    "DATA_DIR = \"/content/drive/MyDrive/SurveyPlan AI/runs/donut_tuning\"\n",
    "metadata_path = os.path.join(DATA_DIR, \"metadata.jsonl\")\n",
    "backup_path = os.path.join(DATA_DIR, \"metadata_backup.jsonl\")\n",
    "\n",
    "if not os.path.exists(metadata_path):\n",
    "    print(f\"File not found: {metadata_path}\")\n",
    "else:\n",
    "    # Create a quick backup just in case\n",
    "    !cp \"{metadata_path}\" \"{backup_path}\"\n",
    "    \n",
    "    # Read current metadata\n",
    "    with open(metadata_path, 'r', encoding='utf-8') as f:\n",
    "        lines = f.readlines()\n",
    "        \n",
    "    new_lines = []\n",
    "    labeled_count = 0\n",
    "    \n",
    "    print(\"Scanning metadata for blank templates...\")\n",
    "    for line in tqdm(lines, desc=\"Auto-Labeling\"):\n",
    "        record = json.loads(line)\n",
    "        gt = record.get(\"ground_truth\", \"\")\n",
    "        \n",
    "        # Check if it's the blank template exported by YOLO\n",
    "        if '{\"gt_parse\": {\"text\": \"\"}}' in gt or '\"text\": \"\"' in gt:\n",
    "            img_path = os.path.join(DATA_DIR, record[\"file_name\"])\n",
    "            label = record.get(\"label\", \"general\")\n",
    "            \n",
    "            if os.path.exists(img_path):\n",
    "                # 1. Determine Prompt (Schema root)\n",
    "                task_prompt = \"<s_general>\"\n",
    "                if \"curve_data\" in label:\n",
    "                    task_prompt = \"<s_lot_geometry>\"\n",
    "                elif \"parcel_info\" in label or \"lot_id\" in label:\n",
    "                    task_prompt = \"<s_parcel_info>\"\n",
    "                elif \"table\" in label:\n",
    "                    task_prompt = \"<s_tabular_data>\"\n",
    "                elif \"dist\" in label or \"az\" in label or \"rad\" in label or \"arc\" in label:\n",
    "                    task_prompt = \"<s_lot_geometry>\"\n",
    "                    \n",
    "                # 2. Run Inference\n",
    "                try:\n",
    "                    img = Image.open(img_path).convert(\"RGB\")\n",
    "                    pred_seq, pred_json = run_direct_inference(img, task_prompt)\n",
    "                    \n",
    "                    # 3. Format Prediction into Ground Truth string\n",
    "                    # Assuming pred_json directly resembles {\"general\": {\"text\": \"123\"}}\n",
    "                    new_gt = {\"gt_parse\": pred_json}\n",
    "                    record[\"ground_truth\"] = json.dumps(new_gt, ensure_ascii=False)\n",
    "                    labeled_count += 1\n",
    "                except Exception as e:\n",
    "                    print(f\"Failed to infer {img_path}: {e}\")\n",
    "                    \n",
    "        new_lines.append(json.dumps(record, ensure_ascii=False) + \"\\n\")\n",
    "        \n",
    "    # Write back the new annotated metadata\n",
    "    with open(metadata_path, 'w', encoding='utf-8') as f:\n",
    "        f.writelines(new_lines)\n",
    "        \n",
    "    print(f\"\\n[SUCCESS] Auto-Labeled {labeled_count} new images using the Donut Model!\")\n",
    "    print(\"Open Google Drive, review the JSON file to fix minor mistakes, and you are ready to train again!\")\n"
  ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook exactly match metric successfully added.")
