from PIL import Image, ImageDraw
import math

def test_quad():
    # create a black image 200x200
    img = Image.new("RGB", (200, 200), "black")
    draw = ImageDraw.Draw(img)
    # Draw a rotated white rectangle in the center
    # Let's say it's 100 wide and 50 tall, rotated by some angle
    # Actually, we can just define 4 points
    tl = [50, 50]
    tr = [150, 50]
    br = [150, 100]
    bl = [50, 100]
    draw.polygon([tuple(tl), tuple(tr), tuple(br), tuple(bl)], fill="white")
    # Add a red mark at TL
    draw.ellipse([tl[0]-5, tl[1]-5, tl[0]+5, tl[1]+5], fill="red")
    img.save("test_original.png")

    w = 100
    h = 50
    # The current logic uses TL, BL, BR, TR
    quad_data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
    crop = img.transform((w, h), Image.QUAD, data=quad_data, resample=Image.BICUBIC)
    crop.save("test_crop.png")

    print("Success")

if __name__ == "__main__":
    test_quad()
