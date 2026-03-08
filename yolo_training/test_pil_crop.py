from PIL import Image, ImageDraw
import json
import math

# We want to see what happens when we crop the exact coordinates from the JSON
def visualize_pil_crop():
    with open('test_corners.json', 'r') as f:
        data = json.load(f)
        
    for idx in range(3):
        item = data[idx]
        corners = item['corners']
        tl, tr, br, bl = corners
        
        # Calculate W and H
        w = int(math.hypot(tr[0]-tl[0], tr[1]-tl[1]))
        h = int(math.hypot(bl[0]-tl[0], bl[1]-tl[1]))
        
        # Find min/max to establish an origin
        min_x = min([p[0] for p in corners])
        max_x = max([p[0] for p in corners])
        min_y = min([p[1] for p in corners])
        max_y = max([p[1] for p in corners])
        
        # Create a blank image large enough to hold the box
        img_w = int(max_x - min_x) + 100
        img_h = int(max_y - min_y) + 100
        
        img = Image.new('RGB', (img_w, img_h), 'black')
        draw = ImageDraw.Draw(img)
        
        # Translate corners relative to this new image
        def to_local(pt):
            return (pt[0] - min_x + 50, pt[1] - min_y + 50)
            
        local_tl = to_local(tl)
        local_tr = to_local(tr)
        local_br = to_local(br)
        local_bl = to_local(bl)
        
        # Draw the polygon in white
        draw.polygon([local_tl, local_tr, local_br, local_bl], fill='white')
        
        # Draw colored corners so we can see what they are
        r = 5
        draw.ellipse([local_tl[0]-r, local_tl[1]-r, local_tl[0]+r, local_tl[1]+r], fill='red') # TL
        draw.ellipse([local_tr[0]-r, local_tr[1]-r, local_tr[0]+r, local_tr[1]+r], fill='green') # TR
        draw.ellipse([local_br[0]-r, local_br[1]-r, local_br[0]+r, local_br[1]+r], fill='blue') # BR
        draw.ellipse([local_bl[0]-r, local_bl[1]-r, local_bl[0]+r, local_bl[1]+r], fill='yellow') # BL
        
        img.save(f'box_{idx}_original.png')
        
        # NOW CROP IT EXACTLY LIKE PIPELINE.PY
        # quad_data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
        quad_data = (
            local_tl[0], local_tl[1],
            local_bl[0], local_bl[1],
            local_br[0], local_br[1],
            local_tr[0], local_tr[1]
        )
        
        crop = img.transform((w, h), Image.QUAD, data=quad_data, resample=Image.BICUBIC)
        crop.save(f'box_{idx}_crop.png')
        print(f"Saved box_{idx}_crop.png")

if __name__ == "__main__":
    visualize_pil_crop()
