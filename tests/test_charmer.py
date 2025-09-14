import pytest 
from textattack.shared.attacked_text import AttackedText
from textattack.transformations import word_swap_charmer

from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.goal_functions.classification import UntargetedClassification

# # import and wrap a small model
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from textattack.models.wrappers import HuggingFaceModelWrapper

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


# fixture and setup for the tests
@pytest.fixture
def example_text():
    return AttackedText("hello world")

@pytest.fixture
def charmer_transformation():
    return word_swap_charmer.WordSwapCharmer()

@pytest.fixture
def charmer_goal_function():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    return UntargetedClassification(wrapper)


# test transformations
def test_charmer_word_swap(example_text):
    transformation = word_swap_charmer.WordSwapCharmer()
    transformed_texts = transformation(current_text=example_text, indices_to_modify=[0, 1])
    
    assert isinstance(transformed_texts, list)
    for transformed in transformed_texts:
        # assert isinstance(transformed, str)
        assert isinstance(transformed, AttackedText)
        
        # check each transformed text has one character changed
        # print(transformed)
        
    

# test search methods
def test_charmer_search_method(example_text, charmer_transformation, charmer_goal_function):
    from textattack.search_methods import charmer
    search_method = charmer.Charmer()
    
    # attach the neccessary component for the search method to work
    search_method.goal_function = charmer_goal_function
    search_method.transformation = charmer_transformation
    search_method.constraints = []
    
    # Patch the functions needed in search method
    search_method.get_transformations = lambda attacked_text, indices_to_modify: search_method.transformation(
        attacked_text,
        indices_to_modify=indices_to_modify
    )
    search_method.get_goal_results = lambda texts: charmer_goal_function.get_results(texts)
    search_method.filter_transformations = lambda transformations: transformations

    
    # setup initial result
    initial_result, _ = charmer_goal_function.init_attack_example(example_text, ground_truth_output=0)
    
    # call the search method
    final_result = search_method(initial_result)
    
    assert final_result.goal_status is not None
    assert isinstance(final_result.attacked_text.text, str)
    
    

# test integration with attacks
def test_charmer_attack_recipe(example_text, charmer_transformation, charmer_goal_function):
    # from textattack.attack_recipes import charmer_2024
    from textattack import Attack
    from textattack.search_methods import charmer
    from textattack.goal_function_results import GoalFunctionResultStatus
    from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

    attack = Attack(
        goal_function=charmer_goal_function,
        constraints=[],
        transformation=charmer_transformation,
        search_method=charmer.Charmer(max_k=10, candidate_pos=10)
    )
    
    result = attack.attack(example_text, 1) # a dummy ground truth label
    
    assert result is not None
    if isinstance(result, (FailedAttackResult, SuccessfulAttackResult)):
        assert result.perturbed_result.goal_status in (
            GoalFunctionResultStatus.SUCCEEDED,
            GoalFunctionResultStatus.SEARCHING,
        )
    elif isinstance(result, SkippedAttackResult):
        # Handle skipped attacks if necessary
        pass
    
    



