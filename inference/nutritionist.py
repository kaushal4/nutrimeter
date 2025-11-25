import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
class Nutritionist:
    """
    Uses DeepSeek LLM to generate a nutritionist-style assessment
    based on the analyzed meal data and scene context.
    """
    
    def __init__(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not api_key:
            print("WARNING: DEEPSEEK_API_KEY not found. Nutritionist feature will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print("Nutritionist (DeepSeek) initialized.")

    def generate_feedback(self, nutrition_data: dict, scene_items: list) -> str:
        """
        Generates a structured nutritional assessment.
        
        Args:
            nutrition_data (dict): The JSON output from the pipeline containing
                                   detected objects and total nutrition.
            scene_items (list): Additional items found by the VLM.
            
        Returns:
            str: The markdown-formatted feedback from the nutritionist.
        """
        if not self.client:
            return "Nutritionist AI is not configured (missing API key)."

        # Prepare the context
        detected_objects = nutrition_data.get("objects_detected", [])
        total_nutrition = nutrition_data.get("total_meal_nutrition_est", {})
        
        # Format detected items for the prompt
        items_str = ""
        for obj in detected_objects:
            name = obj['class_name']
            nutri = obj.get('nutrition_per_100g')
            if nutri:
                # Summarize key macros for this item
                cal = nutri.get('Energy_kcal', 0)
                prot = nutri.get('Protein_g', 0)
                fat = nutri.get('Total lipid (fat)_g', 0)
                carb = nutri.get('Carbohydrate, by difference_g', 0)
                items_str += f"- {name}: {cal}kcal, P:{prot}g, F:{fat}g, C:{carb}g\n"
            else:
                items_str += f"- {name}: (No nutrition data)\n"
                
        if scene_items:
            items_str += f"\nAdditional items observed in scene: {', '.join(scene_items)}"

        # Format total nutrition
        total_str = "Total Estimated Nutrition:\n"
        if "per_100g" in total_nutrition:
            t = total_nutrition["per_100g"]
            total_str += (
                f"Calories: {t.get('Energy_kcal', 0)}\n"
                f"Protein: {t.get('Protein_g', 0)}g\n"
                f"Fat: {t.get('Total lipid (fat)_g', 0)}g\n"
                f"Carbs: {t.get('Carbohydrate, by difference_g', 0)}g\n"
            )
        else:
            total_str += "Could not calculate total nutrition.\n"

        system_prompt = (
            "You are a professional, empathetic, and science-based Nutritionist. "
            "Your goal is to provide a brief, actionable assessment of a client's meal. "
            "Do NOT use phrases like 'Here is your analysis' or 'As a nutritionist'. "
            "Just dive straight into the content. "
            "Format your response in Markdown."
        )

        user_prompt = (
            f"Analyze the following meal:\n\n"
            f"{items_str}\n\n"
            f"{total_str}\n\n"
            "Provide a response covering exactly these 4 points:\n\n"
            "1. **Macronutrient Split Assessment**: Break down the 'Big Three' (Protein, Fiber/Carbs, Fats). "
            "Check for complete protein (goal 20-30g), complex vs simple carbs, and healthy vs inflammatory fats.\n\n"
            "2. **Micronutrient Density Scan**: Scan for 'colors' and food groups. Are these 'empty' calories?\n\n"
            "3. **Satiety Index Evaluation**: Predict how long this meal will keep the client full based on fiber/protein.\n\n"
            "4. **The 'Praise-Correct-Prescribe' Feedback**: \n"
            "   - **Step A (Praise)**: Validate one good thing.\n"
            "   - **Step B (Prioritize)**: Identify the single biggest lever for improvement.\n"
            "   - **Step C (Actionable Swap)**: Give a specific alternative.\n"
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating nutritionist feedback: {str(e)}"
