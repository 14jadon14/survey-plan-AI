
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Union, Any

class DocumentParser:
    def __init__(self, model_path: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        """
        Initialize the DocumentParser with a Donut model.
        
        Args:
            model_path (str): HeightFace model hub path or local path to the Donut model.
                              Defaults to 'naver-clova-ix/donut-base-finetuned-cord-v2'.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Donut model from {model_path} on {self.device}...")
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def process_image(self, image: Union[str, Image.Image], bboxes: List[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Process an image (and optional bounding boxes) to extract text using Donut.

        Args:
            image (str or PIL.Image.Image): Path to image or PIL Image object.
            bboxes (list of lists, optional): List of [xmin, ymin, xmax, ymax] coordinates.
                                              If None, processes the entire image.

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
            for bbox in bboxes:
                # Ensure bbox format is correct and within image bounds
                try:
                    # Basic validation and cropping
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
        task_prompt = "<s_cord-v2>" # Specific to CORD finetuning, might explicitly fail on base if token not in vocab.
        
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
