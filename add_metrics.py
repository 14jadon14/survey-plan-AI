import json

notebook_path = r'c:\Users\Jadon\Desktop\SurveyPlan AI\document_understanding\finetune_donut.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # Target the inference evaluation cell
        if "char_dist = editdistance.eval(pred_seq, target_sequence)" in src and "INFERENCE EVALUATION" in src:
            
            # Find the index to slice and replace the evaluation part
            lines = cell['source']
            for i, line in enumerate(lines):
                if line.startswith("char_dist = editdistance.eval"):
                    slice_idx = i
                    break
            
            new_lines = lines[:slice_idx]
            
            new_metrics = [
                "char_dist = editdistance.eval(pred_seq, target_sequence)\n",
                "cer_percentage = (char_dist / max(len(target_sequence), 1)) * 100\n",
                "exact_match_seq = (pred_seq == target_sequence)\n",
                "\n",
                "def count_exact_matches(gt, pred):\n",
                "    if not isinstance(gt, dict) or not isinstance(pred, dict): return 0, 1\n",
                "    matches = 0\n",
                "    total = 0\n",
                "    for k, v in gt.items():\n",
                "        if isinstance(v, dict):\n",
                "            m, t = count_exact_matches(v, pred.get(k, {}))\n",
                "            matches += m\n",
                "            total += t\n",
                "        elif isinstance(v, list):\n",
                "            # Simplified list (table) checking\n",
                "            total += len(v)\n",
                "            if k in pred and isinstance(pred[k], list):\n",
                "                for i in range(min(len(v), len(pred[k]))):\n",
                "                    m, t = count_exact_matches(v[i], pred[k][i])\n",
                "                    # Add 1 if the whole row dictionary matched perfectly\n",
                "                    if m == t and t > 0: matches += 1\n",
                "        else:\n",
                "            total += 1\n",
                "            if k in pred and str(pred[k]).strip() == str(v).strip():\n",
                "                matches += 1\n",
                "    return matches, total\n",
                "\n",
                "# Unwrap gt_parse for accurate field comparison\n",
                "try:\n",
                "    gt_inner = gt_raw.get('gt_parse', gt_raw)\n",
                "    # Donut model sometimes predicts an extra wrapper, sometimes just the direct dict\n",
                "    # the processor output is usually just the dict\n",
                "    field_matches, field_total = count_exact_matches(gt_inner, pred_json)\n",
                "    field_acc = (field_matches / max(field_total, 1)) * 100\n",
                "except Exception:\n",
                "    field_acc = 0.0\n",
                "    field_matches = 0\n",
                "    field_total = 1\n",
                "\n",
                "print(\"\\n\" + \"=\"*50)\n",
                "print(\"INFERENCE EVALUATION\")\n",
                "print(\"=\"*50)\n",
                "print(f\"Task Prompt: {task_prompt}\\n\")\n",
                "print(\"--- EXPECTED (Ground Truth) ---\")\n",
                "print(json.dumps(gt_raw, indent=2))\n",
                "print(\"\\n--- PREDICTED ---\")\n",
                "print(json.dumps(pred_json, indent=2))\n",
                "\n",
                "print(\"\\n\" + \"=\"*50)\n",
                "print(\"STATISTICS\")\n",
                "print(\"=\"*50)\n",
                "print(f\"Character Error Rate (CER): {cer_percentage:.2f}%\")\n",
                "print(f\"Sequence Exact Match:       {'Yes ✅' if exact_match_seq else 'No ❌'}\")\n",
                "print(f\"Field-Level Exact Match:    {field_matches}/{field_total} fields correct ({field_acc:.1f}%)\")\n"
            ]
            
            cell['source'] = new_lines + new_metrics

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook exactly match metric successfully added.")
