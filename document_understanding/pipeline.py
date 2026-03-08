
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Union, Any
import numpy as np
import cv2
try:
    from deskew import determine_skew
except ImportError:
    determine_skew = None
    print("Warning: 'deskew' library not found. Text deskewing will be disabled. Install with 'pip install deskew'.")

class DocumentParser:
    def __init__(self, model_path: str = "naver-clova-ix/donut-base"):
        """
        Initialize the DocumentParser with a Donut model.
        
        Args:
            model_path (str): HuggingFace model hub path or local path to the Donut model.
                              Defaults to 'naver-clova-ix/donut-base'.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Donut model from {model_path} on {self.device}...")
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def process_image(self, image: Union[str, Image.Image], bboxes: List[List[int]] = None, corners_list: List[List[List[float]]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Process an image (and optional bounding boxes or corners) to extract text using Donut.

        Args:
            image (str or PIL.Image.Image): Path to image or PIL Image object.
            bboxes (list of lists, optional): List of [xmin, ymin, xmax, ymax] coordinates.
            corners_list (list of lists, optional): List of 4 points [[x,y], [x,y], [x,y], [x,y]] (TL, TR, BR, BL).


        Returns:
            List of dictionaries containing 'bbox', 'parsed_content', and 'crop'.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image}: {e}")
                return []
                
        # Apply explicit deskewing globally to strictly horizontalize document BEFORE cropping
        if determine_skew is not None:
            image, bboxes, corners_list = self.global_deskew(image, bboxes, corners_list)
        
        results = []
        if bboxes or corners_list:
            loop_list = corners_list if corners_list else bboxes
            for i in range(len(loop_list)):
                bbox = bboxes[i] if bboxes and i < len(bboxes) else None
                corners = corners_list[i] if corners_list and i < len(corners_list) else None
                try:
                    if corners:
                        import math
                        tl, tr, br, bl = corners
                        w = int(math.hypot(tr[0]-tl[0], tr[1]-tl[1]))
                        h = int(math.hypot(bl[0]-tl[0], bl[1]-tl[1]))
                        
                        # PIL Image.QUAD takes points in order: TL, BL, BR, TR
                        quad_data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
                        crop = image.transform((w, h), Image.QUAD, data=quad_data, resample=Image.BICUBIC)
                    elif bbox:
                        crop = image.crop(bbox)
                    else:
                        continue
                    
                    parsed_content = self.parse_crop(crop)
                    results.append({"bbox": bbox, "corners": corners, "parsed_content": parsed_content, "crop": crop.copy()})
                except Exception as e:
                    print(f"Error processing item at index {i}: {e}")
                    results.append({"bbox": bbox, "corners": corners, "error": str(e), "crop": None})
        else:
            parsed_content = self.parse_crop(image)
            results.append({"bbox": None, "parsed_content": parsed_content, "crop": image})
            
        return results
        
    def global_deskew(self, image: Image.Image, bboxes: List[List[int]], corners_list: List[List[List[float]]]) -> tuple:
        """
        Detects global document skew angle and rotates both the full page image and all bounding geometries.
        
        Args:
            image (PIL.Image.Image): The full document image.
            bboxes (List[List[int]]): List of bboxes.
            corners_list (List[List[List[float]]]): List of corners.
            
        Returns:
            tuple: (deskewed_image, rotated_bboxes, rotated_corners)
        """
        try:
            # Convert PIL to openCV Grayscale array
            cv_img = np.array(image.convert('L'))
            
            # Determine angle using Hough Transform logic on the ENTIRE page
            angle = determine_skew(cv_img)
            
            if angle is None or abs(angle) < 0.5:
                return image, bboxes, corners_list
                
            # Perform Affine Rotation per user spec
            (h, w) = cv_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Use original RGB for rotation to preserve colors/quality
            rgb_arr = np.array(image)
            rotated_img = cv2.warpAffine(
                rgb_arr, 
                M, 
                (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
            image_out = Image.fromarray(rotated_img)
            
            # Transform global coordinates using the exact same matrix
            def rotate_pt(x, y):
                new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                return [new_x, new_y]
                
            rotated_bboxes = None
            if bboxes:
                rotated_bboxes = []
                for box in bboxes:
                    pts = [
                        rotate_pt(box[0], box[1]),
                        rotate_pt(box[2], box[1]),
                        rotate_pt(box[2], box[3]),
                        rotate_pt(box[0], box[3])
                    ]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    rotated_bboxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
            else:
                rotated_bboxes = bboxes
                    
            rotated_corners = None
            if corners_list:
                rotated_corners = []
                for corners in corners_list:
                    rotated_corners.append([rotate_pt(pt[0], pt[1]) for pt in corners])
            else:
                rotated_corners = corners_list
                    
            return image_out, rotated_bboxes, rotated_corners
            
        except Exception as e:
            print(f"Warning: Global deskewing failed, proceeding with original data. Error: {e}")
            return image, bboxes, corners_list

    def parse_crop(self, crop: Image.Image) -> str:
        """
        Run Donut inference on a single image crop.
        
        Args:
            crop (PIL.Image.Image): Image crop to process.
            
        Returns:
            str: The parsed text/sequence.
        """
        # Prepare inputs
        pixel_values = self.processor(crop, return_tensors="pt").pixel_values.to(self.device)
        
        # Prepare decoder inputs
        # We act as if we are starting generation. 
        # For base model, we can start with the decoder_start_token_id if available,
        # otherwise use a standard token.
        task_prompt = "<s_synthdog>" # Synthetic OCR basic reading task
        
        # Check if task prompt token exists in tokenizer, otherwise use default start token
        if task_prompt in self.processor.tokenizer.get_vocab():
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        else:
            # Fallback to default decoder start token
            decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
            if decoder_start_token_id is None:
                decoder_start_token_id = self.model.config.decoder.bos_token_id
            
            decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=self.device)


        # Generate
        max_length = getattr(self.model.config, "max_position_embeddings", None)
        if max_length is None:
            max_length = getattr(self.model.config.decoder, "max_position_embeddings", 512)
            
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            # early_stopping=True, # Removed to avoid warning
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
        # Decode
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        
        # Cleanup
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        # Remove the start token(s)
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        return sequence

if __name__ == "__main__":
    # Simple test block
    print("Testing existing setup...")
    try:
        parser = DocumentParser() # This will attempt to download/load the model
        print("Parser initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize parser: {e}")
