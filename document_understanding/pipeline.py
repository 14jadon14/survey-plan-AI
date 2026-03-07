
import torch
import math
import re
import cv2
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Union, Any

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

    def process_image(self, image: Union[str, Image.Image], bboxes: List[List[int]] = None, angles: List[float] = None, rect_ws: List[float] = None, rect_hs: List[float] = None) -> List[Dict[str, Any]]:
        """
        Process an image (and optional bounding boxes) to extract text using Donut.

        Args:
            image (str or PIL.Image.Image): Path to image or PIL Image object.
            bboxes (list of lists, optional): List of [xmin, ymin, xmax, ymax] coordinates.
                                              If None, processes the entire image.
            angles (list of floats, optional): List of rotation angles corresponding to each bbox.
                                               Used to deskew the crops before processing.

        Returns:
            List of dictionaries containing 'bbox' and 'parsed_content'.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image}: {e}")
                return []
        
        results = []
        if bboxes:
            for i, bbox in enumerate(bboxes):
                # Ensure bbox format is correct and within image bounds
                try:
                    # Determine angle for this bbox
                    angle = angles[i] if angles and i < len(angles) and angles[i] else 0
                    
                    # Basic validation                    # Determine explicit dimensions if available
                    rect_w = rect_ws[i] if rect_ws and i < len(rect_ws) and rect_ws[i] else 0
                    rect_h = rect_hs[i] if rect_hs and i < len(rect_hs) and rect_hs[i] else 0
                    
                    if rect_w and rect_h:
                        # Deskew using a four-point warp transform based on exact angle and dimensions
                        # This extracts exactly the OBB region and orientates it perfectly horizontally
                        img_arr = np.array(image)
                        xmin, ymin, xmax, ymax = bbox
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        
                        # 1. Define the 4 corners of an unrotated box centered at origin
                        hw = rect_w / 2.0
                        hh = rect_h / 2.0
                        # Order: top-left, top-right, bottom-right, bottom-left
                        base_corners = np.array([
                            [-hw, -hh],
                            [ hw, -hh],
                            [ hw,  hh],
                            [-hw,  hh]
                        ], dtype="float32")
                        
                        # 2. Rotate by the geometric angle
                        angle_rad = math.radians(angle)
                        cos_a = math.cos(angle_rad)
                        sin_a = math.sin(angle_rad)
                        R = np.array([
                            [cos_a, -sin_a],
                            [sin_a,  cos_a]
                        ], dtype="float32")
                        
                        # 3. Apply rotation and translate to the true center point
                        src_pts = np.dot(base_corners, R.T) + np.array([cx, cy], dtype="float32")
                        
                        # 4. Define the destination points for the perfect horizontal crop
                        dst_pts = np.array([
                            [0, 0],
                            [rect_w - 1, 0],
                            [rect_w - 1, rect_h - 1],
                            [0, rect_h - 1]
                        ], dtype="float32")
                        
                        # Perform warp
                        M_warp = cv2.getPerspectiveTransform(src_pts.astype("float32"), dst_pts)
                        warped = cv2.warpPerspective(img_arr, M_warp, (int(rect_w), int(rect_h)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                        
                        if warped.size > 0:
                            crop = Image.fromarray(warped)
                        else:
                            crop = image.crop(bbox) # fallback if empty
                    else:
                        crop = image.crop(bbox)
                    
                    parsed_content = self.parse_crop(crop)
                    results.append({"bbox": bbox, "parsed_content": parsed_content})
                except Exception as e:
                    print(f"Error processing bbox {bbox}: {e}")
                    results.append({"bbox": bbox, "error": str(e)})
        else:
            parsed_content = self.parse_crop(image)
            results.append({"bbox": None, "parsed_content": parsed_content})
            
        return results

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
