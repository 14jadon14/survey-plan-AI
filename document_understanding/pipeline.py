
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
        
        results = []
        if bboxes or corners_list:
            loop_list = corners_list if corners_list else bboxes
            for i in range(len(loop_list)):
                bbox = bboxes[i] if bboxes and i < len(bboxes) else None
                corners = corners_list[i] if corners_list and i < len(corners_list) else None
                try:
                    if corners:
                        import math
                        import numpy as np
                        import cv2
                        
                        tl, tr, br, bl = corners
                        w = int(math.hypot(tr[0]-tl[0], tr[1]-tl[1]))
                        h = int(math.hypot(bl[0]-tl[0], bl[1]-tl[1]))
                        
                        # Use explicit OpenCV perspective transform to absolutely guarantee corner coordinate mapping
                        # PIL Image.QUAD is backwards and brittle (maps destination to source).
                        src_pts = np.array([tl, tr, br, bl], dtype="float32")
                        dst_pts = np.array([
                            [0, 0],       # TL
                            [w - 1, 0],   # TR
                            [w - 1, h - 1], # BR
                            [0, h - 1]    # BL
                        ], dtype="float32")
                        
                        # Calculate perspective matrix
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        
                        # Convert PIL to CV2
                        cv_image = np.array(image)
                        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                            # Convert RGB to BGR for cv2
                            cv_image = cv_image[:, :, ::-1]
                            
                        # Perform warp
                        warped_cv = cv2.warpPerspective(cv_image, matrix, (w, h))
                        
                        # Convert back to PIL
                        if len(warped_cv.shape) == 3 and warped_cv.shape[2] == 3:
                            warped_cv = warped_cv[:, :, ::-1] # BGR to RGB
                        crop = Image.fromarray(warped_cv)
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
