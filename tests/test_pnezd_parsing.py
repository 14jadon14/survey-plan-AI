import pandas as pd
from io import StringIO
import os

def test_pnezd_logic(text, filename="test.txt"):
    print(f"\n--- Testing File: {filename} ---")
    lines = text.strip().split('\n')
    if not lines:
        print("File is empty.")
        return

    first_line = lines[0]
    if ',' in first_line:
        sep = ','
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = None

    # Robust header detection: 
    # Only treat as header if the second and third columns are NOT numeric in the first line, 
    # but ARE numeric in the second line (if it exists).
    first_line_parts = [p.strip().strip('"').strip("'") for p in (first_line.split(sep) if sep else first_line.split())]
    
    def is_numeric(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    # PNEZD typically has Northing and Easting in indices 1 and 2
    line1_coords_numeric = any(is_numeric(first_line_parts[i]) for i in range(1, min(len(first_line_parts), 4)))
    
    is_header = not line1_coords_numeric and len(lines) > 1
    df = pd.read_csv(StringIO(text), sep=sep, engine='python', header=0 if is_header else None)
    
    mapping = {}
    cols = [str(c).lower() for c in df.columns]
    
    # 1. Map Point ID
    for i, c in enumerate(cols):
        if any(k == c for k in ['p', 'pt', 'id', 'name', 'number', 'point']): mapping['point'] = i; break
    if 'point' not in mapping:
        for i, c in enumerate(cols):
            if any(k in c for k in ['point', 'pt', 'id', 'name', 'num']) and not any(k in c for k in ['north', 'east', 'elev']):
                mapping['point'] = i; break

    # 2. Map Northing/Y
    for i, c in enumerate(cols):
        if i in mapping.values(): continue
        if any(k == c for k in ['n', 'y', 'northing', 'north']): mapping['y'] = i; break
    if 'y' not in mapping:
         for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if ('north' in c or 'nort' in c) and 'name' not in c:
                mapping['y'] = i; break
    
    # 3. Map Easting/X
    for i, c in enumerate(cols):
        if i in mapping.values(): continue
        if any(k == c for k in ['e', 'x', 'easting', 'east']): mapping['x'] = i; break
    if 'x' not in mapping:
        for i, c in enumerate(cols):
            if i in mapping.values(): continue
            if ('east' in c or 'east' in c) and 'elev' not in c:
                mapping['x'] = i; break
            
    # 4. Map Elevation/Z
    for i, c in enumerate(cols):
        if i in mapping.values(): continue
        if any(k == c for k in ['z', 'elev', 'height', 'elevation']): mapping['z'] = i; break

    # 5. Map Description
    for i, c in enumerate(cols):
        if i in mapping.values(): continue
        if any(k in c for k in ['desc', 'code', 'note', 'remark', 'info']): mapping['desc'] = i; break

    # Fallback
    if 'point' not in mapping and df.shape[1] >= 1: mapping['point'] = 0
    if 'y' not in mapping and df.shape[1] >= 2: mapping['y'] = 1
    if 'x' not in mapping and df.shape[1] >= 3: mapping['x'] = 2
    if 'z' not in mapping and df.shape[1] >= 4: mapping['z'] = 3
    if 'desc' not in mapping and df.shape[1] >= 5: mapping['desc'] = 4

    print(f"Detected separator: {'whitespace' if sep is None else repr(sep)}")
    print(f"Mapping detected: {mapping}")
    
    for _, row in df.head(2).iterrows():
        p = row.iloc[mapping['point']] if 'point' in mapping else "N/A"
        x = row.iloc[mapping['x']] if 'x' in mapping else "N/A"
        y = row.iloc[mapping['y']] if 'y' in mapping else "N/A"
        print(f"  Point {p}: X={x}, Y={y}")

if __name__ == "__main__":
    # Test 1: Comma delimited with header
    csv_header = "Point,Northing,Easting,Elevation,Description\n1,5000,10000,10,Peg\n2,5010,10010,11,Pipe"
    test_pnezd_logic(csv_header, "comma_header.csv")

    # Test 2: Space delimited without header
    space_no_header = "101 5500.5 12000.7 15.2 START\n102 5520.1 12015.3 15.5 BENDPOINT"
    test_pnezd_logic(space_no_header, "space_no_header.txt")

    # Test 3: Tab delimited with custom headers
    tab_custom = "ID\tY_Coord\tX_Coord\tZ\tNote\n99\t100\t200\t0\tOrigin"
    test_pnezd_logic(tab_custom, "tab_custom.asc")

    # Test 5: User reported failing case (No header, but contains alpha in description)
    user_failing = """1,7439593.1500,2488568.7800,0.0000,NBBDRY
2,7439605.7000,2488593.0400,0.0000,NBBDRY
3,7439621.4900,2488621.4900,0.0000,NBBDRY
9,7439679.3300,2488543.9700,0.0000,
10,7439677.0900,2488565.0800,0.0000,"""
    test_pnezd_logic(user_failing, "user_failing.csv")

    # Test 6: Alphanumeric Point ID
    alphanumeric_id = "8A 7439593.15 2488568.78 0.0 NBBDRY\n8B 7439605.70 2488593.04 0.0 NBBDRY"
    test_pnezd_logic(alphanumeric_id, "alphanumeric.txt")
