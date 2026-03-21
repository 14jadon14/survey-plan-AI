# Labeling Guide: Donut Fine-Tuning for Survey Plans

To train the Donut model effectively, you must follow the correct JSON structure within your `metadata.jsonl` file. Each line should be a self-contained JSON object for one crop.

## 1. Structure of metadata.jsonl
Each line must have this exact format (mind the nested JSON string in `ground_truth`):
```json
{"file_name": "crop_1.jpg", "label": "azimuth", "ground_truth": "{\"gt_parse\": {\"lot_geometry\": {\"az\": \"45-30.15.\", \"dist\": \"\"}}}"}
{"file_name": "crop_2.jpg", "label": "distance", "ground_truth": "{\"gt_parse\": {\"lot_geometry\": {\"az\": \"\", \"dist\": \"123.45\"}}}"}
```

---

## 2. Key "Alphabet" Mapping (Schema)
For each class you label, you must use a standard set of Task Start Tokens and Schema Tags. This ensures the model learns to associate the correct "expertise" with each crop.

> [!TIP]
> **See the full [Alphabet Key](donut_alphabet_key.md) for a complete list of all supported tokens.**

### Examples of Main Schemas:

- **Lot Geometry** (`<s_lot_geometry>`): 
  - `{"gt_parse": {"lot_geometry": {"az": "120-45.30.", "dist": "45.67"}}}`
- **Parcel Info** (`<s_parcel_info>`): 
  - `{"gt_parse": {"parcel_info": {"lot_id": "42", "area_val": "1.25ac"}}}`
- **General Text** (`<s_general>`): 
  - `{"gt_parse": {"general": {"plan_title": "LOT 42 SUBDIVISION", "street": "Main St."}}}`
- **Tabular Data** (`<s_tabular_data>`): 
  - `{"gt_parse": {"tabular_data": {"row": [{"pt_id": "1", "north": "100.0", "east": "200.0"}]}}}`

---

## 3. Normalization Rules (CRITICAL)
Consistency is key to getting a low Character Error Rate (CER). Please follow these rules in your transcriptions:
1.  **Symbols to Dashes**: Replace `°` with `-` (e.g., `45°` -> `45-`).
2.  **Symbols to Dots**: Replace `'` and `"` with `.` (e.g., `30' 15"` -> `30. 15.`).
3.  **No Hallucinations**: Only transcribe what is visibly in the crop. Do not add context that is missing.
4.  **Preserve Whitespace**: If there is a space between numbers, include it (e.g., `120 45` instead of `12045`).

---

## 4. How to Apply Labels
You can open `metadata.jsonl` in a text editor (like VS Code or Notepad) and update the `"ground_truth"` field for each crop. Once you have a batch of 40 samples per class, upload them to your Google Drive and run the `finetune_donut.ipynb` notebook to start training.

**Tip**: If you find that one class is particularly tricky (like "tabular_data"), you can increase the `MAX_CROPS_PER_CLASS` to 100 for just that class to give the model more variety.
