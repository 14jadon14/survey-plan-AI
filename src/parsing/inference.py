import re
import torch

def run_parsing(model, processor, image, device="cpu", task_prompt="<s_survey_parsing>"):
    """
    Runs inference on a single image (crop) and returns the parsed JSON.
    """
    # Prepare input
    # Note: We are doing the preprocessing here inside inference for simplicity of the API,
    # but strictly it could be in processor.py.
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Prepare decoder inputs
    decoder_input_ids = None
    if task_prompt:
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    try:
        # If decoder_input_ids is None, generate will usually use the decoder_start_token_id from config
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return None

    # Decode
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    try:
        return processor.token2json(sequence)
    except Exception as e:
        print(f"[WARN] Failed to convert token sequence to JSON: {e}")
        return {"raw_text": sequence}
