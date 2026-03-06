
import torch
import re
import cv2
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Union, Any

class DocumentParser:
    def __init__(self, model_path: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        """
        Initialize the DocumentParser with a Donut model.
        
        Args:
            model_path (str): HeightFace model hub path or local path to the Donut model.
                              Defaults to 'naver-clova-ix/donut-base-finetuned-docvqa'.
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
                    
                    if angle and angle != 0 and rect_w and rect_h:
                        # Deskew using a four-point warp transform (warpPerspective)
                        # This extracts exactly the OBB region without adjacent noise
                        img_arr = np.array(image)
                        xmin, ymin, xmax, ymax = bbox
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        
                        # Get 4 corners of the rotated bounding box
                        box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), angle))
                        src_pts = np.array(box, dtype="float32")
                        
                        # Helper to order points: top-left, top-right, bottom-right, bottom-left
                        s = src_pts.sum(axis=1)
                        diff = np.diff(src_pts, axis=1) # y - x
                        
                        rect_pts = np.zeros((4, 2), dtype="float32")
                        rect_pts[0] = src_pts[np.argmin(s)]     # top-left
                        rect_pts[2] = src_pts[np.argmax(s)]     # bottom-right
                        rect_pts[1] = src_pts[np.argmin(diff)]  # top-right
                        rect_pts[3] = src_pts[np.argmax(diff)]  # bottom-left
                        
                        # Compute width and height of the tight crop
                        wA = np.linalg.norm(rect_pts[2] - rect_pts[3]) # bottom-right to bottom-left
                        wB = np.linalg.norm(rect_pts[1] - rect_pts[0]) # top-right to top-left
                        max_w = max(int(wA), int(wB))

                        hA = np.linalg.norm(rect_pts[1] - rect_pts[2]) # top-right to bottom-right
                        hB = np.linalg.norm(rect_pts[0] - rect_pts[3]) # top-left to bottom-left
                        max_h = max(int(hA), int(hB))

                        # Ensure text is horizontal (width > height usually for text boxes)
                        if max_h > max_w:
                            # rotate text orientation 90 degrees to make it horizontal
                            rect_pts = np.array([rect_pts[3], rect_pts[0], rect_pts[1], rect_pts[2]], dtype="float32")
                            max_w, max_h = max_h, max_w

                        dst_pts = np.array([
                            [0, 0],
                            [max_w - 1, 0],
                            [max_w - 1, max_h - 1],
                            [0, max_h - 1]], dtype="float32")
                            
                        M_warp = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                        warped = cv2.warpPerspective(img_arr, M_warp, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                        
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
        task_prompt = "<s_docvqa>" # Specific to DocVQA finetuning
        
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
