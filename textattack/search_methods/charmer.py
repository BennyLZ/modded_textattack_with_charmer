from textattack.goal_function_results.goal_function_result import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.attacked_text import AttackedText


from abc import ABC, abstractmethod


class Charmer(SearchMethod, ABC):
    def __init__(self, max_k = 10, candidate_pos = 10):
        self.max_k = max_k
        self.candidate_pos = candidate_pos
        super().__init__()
        
    def perform_search(self, initial_result):
        # initialize attack text
        attacked_text = initial_result.attacked_text
        ground_truth_output = initial_result.ground_truth_output
        
        # iteratively modify the text until the max_k is reached or the goal is achieved
        for _ in range(1, self.max_k + 1):
            # get the top n positions to modify
            inidices_to_modify = self.get_top_n_positions(attacked_text, ground_truth_output, self.candidate_pos)
            
            # get the transformations for the top n positions
            transformed_texts = self.get_transformations(attacked_text, inidices_to_modify)
            
            # the loss vector for each transformed text
            evaluated_results = []
            losses = []
            
            results, _ = self.get_goal_results(transformed_texts)
            
            for result in results:
                if result.raw_output is None:
                    continue
                logits = result.raw_output
                label = result.ground_truth_output
                loss = self.cw_loss(logits, label)
                evaluated_results.append(result)
                losses.append(loss)
                
                # print(result)
            
            # get the best result from the transformed texts based on the loss vector
            if not losses:
                print("No valid transformed results; skipping iteration.")
                continue
            
            best_idx = losses.index(max(losses))
            best_result = evaluated_results[best_idx]
            attacked_text = best_result.attacked_text
            
            # check if the best result achieves the goal 
            if best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                return best_result
            
        
        # return the initial result if no goal is reached
        return initial_result
        
        
    
    '''
    Get the top n positions to modify based on the initial text
    '''
    def get_top_n_positions(self, attacked_text: AttackedText, ground_truth_output: int, n: int):
        special_character = 'ξ'
        test_character = ' '
        
        # expand the sentence by inserting special characters in between letters
        raw_text = attacked_text.text
        expended_sentence = special_character + special_character.join(raw_text) + special_character
        
        # initialize the lose vector
        losses = []
        position_map = []
        
        for i in range(2 * len(raw_text) + 1):
            if expended_sentence[i] == test_character:
                test_sentence = expended_sentence[:i] + special_character + expended_sentence[i+1:]
            else:
                test_sentence = expended_sentence[:i] + test_character + expended_sentence[i+1:]
                
            # calculate the loss for the test sentence
            contracted_sentence = AttackedText(test_sentence.replace(special_character, ''))
            results, _ = self.get_goal_results([contracted_sentence])
            
            if not results or results[0].raw_output is None:
                # print(f"Empty results for: '{contracted_sentence.text}'")
                continue
            
            logits = results[0].raw_output
            loss = self.cw_loss(logits, ground_truth_output)

            losses.append(loss)
            position_map.append(i)
            
        # index of top n positions in the loss vector
        if not losses:
            return []
        
        top_n_indices = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:n]
        return [position_map[i] for i in top_n_indices]
    

    
    '''
    Implement the Carlini-Wagner Loss for evaluating each transformed text
    '''
    def cw_loss(self, logits, true_label, kappa=0):
        # print("logits:", logits, "type:", type(logits))
        true_logit = logits[true_label]
        other_logits = [logits[i] for i in range(len(logits)) if i != true_label]
        max_other = max(other_logits)
        return max(max_other - true_logit, -kappa)
    
    @property
    def is_black_box(self):
        return False
    
    
        

        
        