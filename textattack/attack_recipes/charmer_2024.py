"""
    Charmer : Revisiting character-level attacks on text classifiers
"""

from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import word_swap_charmer
from textattack.search_methods import charmer
from .attack_recipe import AttackRecipe

class Charmer2024(AttackRecipe):
    def build(model_wrapper, max_k=10, candidate_pos=10):
        # transformation, and no semantic constraints
        transformation = word_swap_charmer.WordSwapCharmer()

        # goal function
        goal_function = UntargetedClassification(model_wrapper)
        
        # search method
        search_method = charmer.Charmer(max_k, candidate_pos) # limit max k and number of candidate position to 3 for faster test
        
        return Attack(goal_function=goal_function, 
                      constraints=[], 
                      transformation=transformation, 
                      search_method=search_method)
        