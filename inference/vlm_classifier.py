import torch
import numpy as np
from PIL import Image
# --- 1. Import BitsAndBytesConfig ---
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import json
import re
from qwen_vl_utils import process_vision_info

class VLMClassifier:
    """
    Loads the local Qwen2-VL-7B-Instruct model using 4-bit quantization
    for very fast loading.
    """
    
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    CROP_SIZE = (336, 336)
    
    def __init__(self):
        """
        Loads the Qwen VLM using 4-bit quantization.
        """
        print(f"Loading VLM model: {self.MODEL_ID} (in 4-bit)")
        
        # --- 2. Define the 4-bit configuration ---
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # --- 3. Load the model with the new config ---
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            quantization_config=quantization_config, # Pass the config object
            device_map="auto",  # Automatically uses your CUDA device
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("Qwen VLM model (4-bit) loaded successfully.")

    def _filter_and_sort_masks(self, image_np: np.ndarray, sam_masks: list, top_n: int) -> list:
        """
        Filters out masks that are too small or too large
        and returns the top_n most likely candidates.
        """
        # Get total image size
        total_pixels = image_np.shape[0] * image_np.shape[1]
        
        # --- Define new, stricter area limits ---
        # Don't include tiny masks (less than 0.5% of image)
        min_area_pixels = total_pixels * 0.005
        
        # Don't include huge masks (more than 15% of image)
        # This will filter out the bowl (ID 1) and the
        # large merged mask (ID 2).
        max_area_pixels = total_pixels * 0.15
        
        # Filter masks based on area
        filtered_masks = [
            m for m in sam_masks 
            if min_area_pixels < m['area'] < max_area_pixels
        ]
        
        # Sort by area, largest first
        sorted_masks = sorted(filtered_masks, key=(lambda x: x['area']), reverse=True)
        
        # Return the top N
        return sorted_masks[:top_n]

    def _create_collage(self, image_np: np.ndarray, masks: list) -> Image.Image:
        # (This function is identical)
        resized_crops = []
        for mask in masks:
            x, y, w, h = mask['bbox']
            crop_np = image_np[y:y+h, x:x+w]
            crop_pil = Image.fromarray(crop_np)
            crop_resized = crop_pil.resize(self.CROP_SIZE, Image.Resampling.LANCZOS)
            resized_crops.append(np.array(crop_resized))
        
        collage_np = np.hstack(resized_crops)
        return Image.fromarray(collage_np)

    def _clean_json_scaffolding(self, text: str) -> str:
        # (This function is identical)
        json_block_match = re.search(r"```json\s*(\{.*\S.*\}|\[.*\S.*\])\s*```", text, re.DOTALL)
        
        if json_block_match:
            return json_block_match.group(1)
        else:
            return text

    def _get_vlm_response(self, collage_img: Image.Image, num_objects: int) -> list:
        # (This function is identical, but includes the ValueError fix)
        prompt = (
            f"This image shows {num_objects} objects in a row. "
            f"Identify each object, from left to right. "
            f"Return *only* a JSON list of their names, like: "
            f"{{\"objects\": [\"name1\", \"name2\", ...]}}"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": collage_img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # --- This is the fix for the ValueError ---
            # Unpack only 2 returned values
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"\n--- VLM RAW RESPONSE ---\n{output_text}\n------------------------")
            
            cleaned_text = self._clean_json_scaffolding(output_text)
            json_match = re.search(r"\{.*\S.*\}", cleaned_text, re.DOTALL)
            
            if not json_match:
                print(f"VLM Warning: No JSON block found in cleaned response.")
                return ["unknown"] * num_objects
                
            json_str = json_match.group(0)
            
            try:
                parsed_json = json.loads(json_str)
                if "objects" in parsed_json and isinstance(parsed_json["objects"], list):
                    return parsed_json["objects"]
                else:
                    print(f"VVLM Warning: JSON in wrong format: {json_str}")
                    return ["unknown"] * num_objects
            except json.JSONDecodeError as e:
                print(f"VLM JSONDecodeError: Failed to parse string: '{json_str}'")
                print(f"Error was: {e}")
                return ["unknown"] * num_objects
                
        except Exception as e:
            print(f"Error during VLM inference: {e}")
            import traceback
            print(traceback.format_exc())
            return ["error"] * num_objects

    def run_classification(self, image_np: np.ndarray, sam_masks: list, top_n: int = 5) -> list:
        # (This function is identical, but includes the zip() fix)
        top_masks = self._filter_and_sort_masks(image_np, sam_masks, top_n)
        if not top_masks:
            return [] 
            
        collage_img = self._create_collage(image_np, top_masks)

        collage_img.save("outputs/debug_collage.png")

        labels = self._get_vlm_response(collage_img, len(top_masks))
        
        matched_objects = []
        # --- This is the fix for the zip() error ---
        for mask, label in zip(top_masks, labels):
            matched_objects.append({
                "class_name": str(label).lower(),
                "confidence": 0.99,
                "area_pixels": int(mask['area']),
                "bbox": [int(b) for b in mask['bbox']],
                "segmentation": mask['segmentation']
            })
            
        return matched_objects