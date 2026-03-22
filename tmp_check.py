import os, json
from pathlib import Path

base_dir = Path('c:/Users/Jadon/Desktop/SurveyPlan AI')

def check_dir(d):
    if not d.exists(): return
    print(f'\n--- {d.name} ---')
    for split in ['train', 'valid', 'validation', 'val']:
        split_dir = d / split
        if split_dir.exists():
            img_dir = split_dir / 'images'
            if img_dir.exists():
                imgs = list(img_dir.glob('*'))
                print(f'{split} images: {len(imgs)}')
            else:
                files = list(split_dir.glob('*'))
                print(f'{split} images dir NOT FOUND (found files: {len(files)})')
            
            json_files = list(split_dir.glob('*.json'))
            for jf in json_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)
                        print(f'  JSON {jf.name}: images={len(data.get("images", []))}, annotations={len(data.get("annotations", []))}')
                except Exception as e:
                    print(f'  JSON {jf.name}: error')

check_dir(base_dir / 'custom_data')
check_dir(base_dir / 'data')
check_dir(base_dir / 'sliced_dataset')
check_dir(base_dir / 'survey_obb_dataset')
