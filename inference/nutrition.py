import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

class NutritionFinder:
    """
    Handles loading the nutrition database and performing semantic search
    to find the best nutritional match for a given food name.
    """
    
    # Use a well-regarded, fast model
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    # Paths for the database and the pre-computed embeddings cache
    DB_PATH = 'assets/nutrition_db.json'
    EMBEDDINGS_PATH = 'assets/db_embeddings.pt'

    def __init__(self):
        """
        Loads the model, the nutrition DB, and pre-computes embeddings.
        """
        print("Initializing NutritionFinder...")
        
        if not os.path.exists(self.DB_PATH):
            print(f"FATAL ERROR: Nutrition database not found at {self.DB_PATH}")
            print("Please run the preprocessing script first.")
            raise FileNotFoundError(self.DB_PATH)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading sentence transformer model '{self.MODEL_NAME}' to {self.device}...")
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)

        # 1. Load the pre-parsed nutrition database
        print(f"Loading nutrition database from {self.DB_PATH}...")
        with open(self.DB_PATH, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
            
        # Get a simple list of all food names
        self.food_names = [item['food_name'] for item in self.db]

        # 2. Load or compute database embeddings
        if os.path.exists(self.EMBEDDINGS_PATH):
            print(f"Loading cached embeddings from {self.EMBEDDINGS_PATH}...")
            self.db_embeddings = torch.load(self.EMBEDDINGS_PATH, map_location=self.device)
        else:
            print("No cached embeddings found. Computing them now (this may take a minute)...")
            self.db_embeddings = self.model.encode(
                self.food_names, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                device=self.device
            )
            print(f"Saving embeddings to {self.EMBEDDINGS_PATH} for next time...")
            os.makedirs(os.path.dirname(self.EMBEDDINGS_PATH), exist_ok=True)
            torch.save(self.db_embeddings, self.EMBEDDINGS_PATH)
            
        print("NutritionFinder is ready.")

    def get_nutrition_profile(self, food_name: str, min_confidence=0.70):
        """
        Finds the nutrition profile for a given food name using semantic search.
        
        Args:
            food_name (str): The name from the VLM (e.g., "hash browns").
            min_confidence (float): The minimum cosine similarity to consider a match.
        
        Returns:
            dict: The nutrition dictionary (per 100g) if a match is found.
            None: If no match is found above the confidence threshold.
        """
        
        # 1. Encode the query
        query_embedding = self.model.encode(
            food_name, 
            convert_to_tensor=True, 
            device=self.device
        )
        
        # 2. Compute cosine similarities (finds similarity between 1 query and all DB items)
        # This is extremely fast on a GPU
        cos_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
        
        # 3. Find the best match
        best_match_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_match_idx].item()
        
        best_match_name = self.food_names[best_match_idx]
        
        # print(f"Query: '{food_name}' | Best Match: '{best_match_name}' | Score: {best_score:.4f}")

        # 4. Return the data only if it meets our threshold
        if best_score >= min_confidence:
            print(f"Nutrition Match: '{food_name}' -> '{best_match_name}' (Score: {best_score:.2f})")
            # Return the full nutrition object for the best match
            return self.db[best_match_idx]['nutrition']
        else:
            print(f"Nutrition Mismatch: '{food_name}' (Best: '{best_match_name}', Score: {best_score:.2f} < {min_confidence})")
            return None