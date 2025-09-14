import re
from textattack.transformations import Transformation
from textattack.shared.attacked_text import AttackedText
import random


class WordSwapCharmer(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        """
        Returns a list of all possible transformations for current_text at the given indices to modify. 
        
        current_text: text to be perturbed
        indices_to_modify: the top n positions to modify in current_text
        """
        transformed_text = []
        
        # define the alphabet
        alphabet = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
            '-', '_', '=', '+', '{', '}', '[', ']', '|', ':',
            ';', '"', "'", '<', '>', ',', '.', '?', '/',
            '`', '~', ' ', 'ξ'
        ]
        special_character = 'ξ'
        expanded_text = special_character + special_character.join(str(current_text.text)) + special_character
        
        # iterate over the indices to modify
        for i in indices_to_modify:
            for j in range(len(alphabet)):
                new_char = alphabet[j]
                text_copy = expanded_text[:i] + new_char + expanded_text[i+1:]
                
                # contract the text back to its original form
                text_copy = text_copy.replace(special_character, '')
                
                transformed_text.append(AttackedText(text_copy))
                
        return transformed_text
                
            
          