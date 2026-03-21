# Donut Alphabet Key: Task & Schema Tokens

This reference sheet tracks the **Task Start Tokens** and the **Schema Tags** required for your `metadata.jsonl` ground truth. Consistency with this "alphabet" ensures the model learns your specific data structures effectively.

---

## 1. Quick Reference Table

| Task Topic | Task Start Token | Content Schema Tags (Tokens) |
| :--- | :--- | :--- |
| **Lot Geometry** | `<s_lot_geometry>` | `<az>`, `<dist>`, `<radius>`, `<arc>`, `<delta>`, `<chord>` |
| **Parcel Info** | `<s_parcel_info>` | `<lot_id>`, `<adj_id>`, `<area_val>` |
| **General Text** | `<s_general>` | `<plan_title>`, `<title_data>`, `<street>`, `<text>` |
| **Tabular Data** | `<s_tabular_data>`| `<row>`, `<pt_id>`, `<north>`, `<east>` |

---

## 2. Usage Examples

### A. Lot Geometry (Azimuths/Distances)
**Prompt**: `<s_lot_geometry>`
**Schema**:
```json
{"lot_geometry": {"az": "120-45.30.", "dist": "145.22"}}
```

### B. Parcel Info
**Prompt**: `<s_parcel_info>`
**Schema**:
```json
{"parcel_info": {"lot_id": "42", "adj_id": "PT. 23", "area_val": "1.25ac"}}
```

### C. General Information
**Prompt**: `<s_general>`
**Schema**:
```json
{"general": {"plan_title": "LOT 42 SUBDIVISION", "street": "Main St."}}
```

---

## 3. Important Notes
1.  **Dynamic Discovery**: Your `finetune_donut.ipynb` script will automatically add any *new* keys you use in your JSON to the model's vocabulary. However, it is best to stick to the tokens above for standard survey features.
2.  **Order Matters**: Always wrap your schema in the corresponding root key (e.g. `lot_geometry`) so the model knows which "Expertise" to activate using the Task Start Token.
3.  **No Spaces in Tags**: Schema tokens cannot have spaces (e.g., use `<plan_title>` instead of `<plan title>`).
