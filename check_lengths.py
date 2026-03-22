import json

def json2token(obj, sort_json_key=True):
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return str(obj["text_sequence"])
        output = ""
        keys = sorted(obj.keys()) if sort_json_key else obj.keys()
        for k in keys:
            output += f"<{k}>"
            output += json2token(obj[k], sort_json_key)
            output += f"</{k}>"
        return output
    elif isinstance(obj, list):
        return "".join([json2token(item, sort_json_key) for item in obj])
    else:
        return str(obj)

def check_lengths(metadata_path):
    max_chars = 0
    max_entry = ""
    lengths = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            gt_raw = json.loads(data['ground_truth'])
            gt_parse = gt_raw.get('gt_parse', gt_raw)
            
            # Simulate prompt + sequence
            root_key = list(gt_parse.keys())[0] if isinstance(gt_parse, dict) else "general"
            inner_gt = gt_parse[root_key] if isinstance(gt_parse, dict) and len(gt_parse)==1 else gt_parse
            
            sequence = f"<s_{root_key}>" + json2token(inner_gt) + "</s>"
            lengths.append(len(sequence))
            
            if len(sequence) > max_chars:
                max_chars = len(sequence)
                max_entry = data['file_name']
                
    lengths.sort(reverse=True)
    print(f"Max characters: {max_chars} (File: {max_entry})")
    print(f"Top 5 character lengths: {lengths[:5]}")
    
    # Rough token estimation: 
    # Custom tags like <row> count as 1 token.
    # Text inside often averages ~2.5 chars per token for numbers/survey data.
    # Let's assume a safe upper bound: tokens = character_count / 1.5 (conservative since many are single-token tags)
    est_tokens = max_chars / 1.5 
    print(f"Estimated max tokens: ~{int(est_tokens)}")
    
    if est_tokens > 768:
        print("\nWARNING: Some entries might exceed 768 tokens!")
    else:
        print("\nSUCCESS: 768 tokens seems sufficient for the current dataset.")

if __name__ == "__main__":
    check_lengths(r"C:\Users\Jadon\Downloads\dataset\metadata.jsonl")
