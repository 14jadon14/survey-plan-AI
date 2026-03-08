from PIL import Image, ImageDraw

def test_quad_transform():
    # Make a dummy 500x500 image with a tilted text
    img = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(img)
    
    # Let's say we have an OBB
    # Center = 250, 250
    # width = 200, height = 50
    # angle = 30 degrees (tilted clockwise, so right side is lower, y increases down)
    
    # I will just define TL, TR, BR, BL manually
    tl = (150, 200)
    tr = (350, 250)
    br = (330, 300)
    bl = (130, 250)
    
    # Draw the OBB
    draw.polygon([tl, tr, br, bl], outline="red", width=2)
    # Draw some text
    draw.line([tl, br], fill="blue", width=2)
    
    img.save("test_original.png")
    
    # We want a tight crop of this OBB, un-rotated.
    # We need the output dimensions:
    import math
    w = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
    h = math.hypot(bl[0]-tl[0], bl[1]-tl[1])
    
    # PIL Quad maps to the standard rectangle (0,0, w,h)
    # The data tuple corresponds to the corners in the original image
    # mapped to top-left, bottom-left, bottom-right, top-right
    # data = (x0, y0, x1, y1, x2, y2, x3, y3) -> (TL_x, TL_y, BL_x, BL_y, BR_x, BR_y, TR_x, TR_y)
    quad_data = (
        tl[0], tl[1],
        bl[0], bl[1],
        br[0], br[1],
        tr[0], tr[1]
    )
    
    # Do the transform
    w_int, h_int = int(round(w)), int(round(h))
    transformed = img.transform((w_int, h_int), Image.QUAD, quad_data, resample=Image.BICUBIC)
    
    transformed.save("test_transformed.png")
    print(f"Original quad: TL={tl}, TR={tr}, BR={br}, BL={bl}")
    print(f"Transformed size: {transformed.size}")
    print("Success")

if __name__ == "__main__":
    test_quad_transform()
