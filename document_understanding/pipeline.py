
import os
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import PreTrainedTokenizerFast
from typing import List, Dict, Union, Any
import numpy as np
import cv2
# Deskewing disabled as per user request
determine_skew = None


def _load_processor_safely(model_path: str) -> DonutProcessor:
    """
    Load DonutProcessor, ensuring the fast tokenizer (tokenizer.json) is used.

    Problem: the fine-tuned model's tokenizer_config.json names
    'XLMRobertaTokenizer' as the tokenizer_class. HuggingFace then tries to
    load the slow sentencepiece-based tokenizer that requires a
    'sentencepiece.bpe.model' file — which is typically absent after
    fine-tuning with add_special_tokens().

    Solution for local paths: bypass the tokenizer_class resolution by
    directly instantiating PreTrainedTokenizerFast from tokenizer.json, then
    attach all special tokens from tokenizer_config.json, and finally build a
    DonutProcessor with the patched tokenizer + the image processor loaded from
    preprocessor_config.json.
    
    For HuggingFace Hub paths (no local tokenizer.json), fall back to the
    standard from_pretrained() call.
    """
    tokenizer_json = os.path.join(model_path, "tokenizer.json")
    tokenizer_cfg  = os.path.join(model_path, "tokenizer_config.json")
    preprocessor_cfg = os.path.join(model_path, "preprocessor_config.json")

    if os.path.isfile(tokenizer_json):
        # ── Local fine-tuned weights path ──────────────────────────────────
        print(f"[INFO] Found local tokenizer.json – loading fast tokenizer directly.")
        
        # 1. Load the fast tokenizer directly from tokenizer.json, skipping
        #    the tokenizer_config.json class resolution.
        import json
        tokenizer_config = {}
        if os.path.isfile(tokenizer_cfg):
            with open(tokenizer_cfg, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

        # Build a PreTrainedTokenizerFast without auto-selecting a slow class
        tok = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json,
            bos_token   = tokenizer_config.get("bos_token",   "<s>"),
            eos_token   = tokenizer_config.get("eos_token",   "</s>"),
            unk_token   = tokenizer_config.get("unk_token",   "<unk>"),
            pad_token   = tokenizer_config.get("pad_token",   "<pad>"),
            sep_token   = tokenizer_config.get("sep_token",   "</s>"),
            cls_token   = tokenizer_config.get("cls_token",   "<s>"),
            mask_token  = tokenizer_config.get("mask_token",  "<mask>"),
            model_max_length = tokenizer_config.get("model_max_length", int(1e30)),
        )

        # Register any extra special tokens that were added during fine-tuning
        extra_tokens = tokenizer_config.get("extra_special_tokens", [])
        if extra_tokens:
            tok.add_special_tokens({"additional_special_tokens": extra_tokens})

        print(f"[INFO] Fast tokenizer loaded. Vocab size: {len(tok)}")

        # 2. Load the image processor separately
        from transformers import DonutImageProcessor
        if os.path.isfile(preprocessor_cfg):
            image_processor = DonutImageProcessor.from_pretrained(model_path)
        else:
            image_processor = DonutImageProcessor()

        # 3. Assemble a DonutProcessor from the two components
        processor = DonutProcessor(image_processor=image_processor, tokenizer=tok)
        return processor
    else:
        # ── HuggingFace Hub / standard path ───────────────────────────────
        print(f"[INFO] No local tokenizer.json found – using from_pretrained().")
        return DonutProcessor.from_pretrained(model_path)


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

        # --- Load Processor (handles custom schema tokens) ---
        self.processor = _load_processor_safely(model_path)
        
        # Optimize size for survey plan crops (middle ground between small labels and large notes)
        # Default is often 2560x1920; 1280x960 provides high fidelity without extreme interpolation.
        self.processor.image_processor.size = {"height": 1280, "width": 960}
        self.processor.image_processor.do_align_long_axis = True
        
        # --- Load Model ---
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)

        # --- Resize embeddings if vocab grew during fine-tuning ---
        # The fine-tuned tokenizer may have more tokens than the base model's
        # embedding matrix. We resize to keep them in sync.
        tokenizer_vocab_size = len(self.processor.tokenizer)
        model_vocab_size = self.model.config.decoder.vocab_size
        if tokenizer_vocab_size != model_vocab_size:
            print(
                f"[INFO] Tokenizer vocab size ({tokenizer_vocab_size}) differs from "
                f"model decoder vocab size ({model_vocab_size}). Resizing embeddings..."
            )
            self.model.decoder.resize_token_embeddings(tokenizer_vocab_size)
            self.model.config.decoder.vocab_size = tokenizer_vocab_size
            # Also update the top-level config used for generation
            if hasattr(self.model.config, "vocab_size"):
                self.model.config.vocab_size = tokenizer_vocab_size

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
                    if bbox:
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
        
        # Use the decoder_start_token_id defined by the model config, which reflects the custom finetuning token task prompt.
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
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
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
    import sys
    # Accept optional path argument: python pipeline.py [model_path]
    test_path = sys.argv[1] if len(sys.argv) > 1 else "naver-clova-ix/donut-base"
    print(f"Testing DocumentParser with model: {test_path}")
    try:
        parser = DocumentParser(model_path=test_path)
        print("Parser initialized successfully.")
        print(f"  Tokenizer vocab size : {len(parser.processor.tokenizer)}")
        print(f"  Decoder start token  : {parser.model.config.decoder_start_token_id}")
        print(f"  Device               : {parser.device}")
    except Exception as e:
        print(f"Failed to initialize parser: {e}")
        sys.exit(1)
