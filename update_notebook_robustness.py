import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # Target the token extraction loop in Section 3
        if "for item in train_dataset:" in src and "try:" not in src:
            new_src = [
                "from tqdm import tqdm\n",
                "import json\n",
                "\n",
                "# Dynamically extract schema tokens from the dataset\n",
                "task_start_tokens = set()\n",
                "schema_tokens = set()\n",
                "\n",
                "def extract_tokens(obj):\n",
                "    if isinstance(obj, dict):\n",
                "        for k, v in obj.items():\n",
                "            schema_tokens.add(f\"<{k}>\")\n",
                "            extract_tokens(v)\n",
                "    elif isinstance(obj, list):\n",
                "        for item in obj:\n",
                "            extract_tokens(item)\n",
                "\n",
                "print(\"Analyzing dataset for tokens...\")\n",
                "for item in tqdm(train_dataset):\n",
                "    try:\n",
                "        gt = item['ground_truth']\n",
                "        if isinstance(gt, str): gt = json.loads(gt)\n",
                "        gt_parse = gt.get('gt_parse', gt)\n",
                "        \n",
                "        if isinstance(gt_parse, dict) and len(gt_parse) == 1:\n",
                "            root_key = list(gt_parse.keys())[0]\n",
                "            task_start_tokens.add(f\"<s_{root_key}>\")\n",
                "            extract_tokens(gt_parse[root_key])\n",
                "    except FileNotFoundError:\n",
                "        # Skip missing files if they weren't caught by pre-processing\n",
                "        continue\n",
                "    except Exception as e:\n",
                "        print(f\"Warning: Skipping an item due to error: {e}\")\n",
                "\n",
                "task_start_tokens = sorted(list(task_start_tokens))\n",
                "schema_tokens = sorted(list(schema_tokens))\n",
                "\n",
                "new_tokens = task_start_tokens + schema_tokens\n",
                "for token in schema_tokens:\n",
                "    new_tokens.append(token.replace(\"<\", \"</\"))\n",
                "\n",
                "processor.tokenizer.add_tokens(new_tokens)\n",
                "print(f\"\\nDynamically discovered {len(task_start_tokens)} task tokens and {len(schema_tokens)} schema tokens!\")"
            ]
            cell['source'] = new_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook successfully updated with safety check.")
