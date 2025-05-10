import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from fuzzy.fuzzy import FuzzyLogicEvaluator

class IdeasFuzzyEvaluator(FuzzyLogicEvaluator):
    
    def __init__(self):
        """Initialize the evidence fuzzy logic system."""
        super().__init__()
        self.control_system = self._create_control_system()

    def _create_control_system(self):
        # Define fuzzy variables
        score = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'score')
        category_score = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'category_score')
    
        # Define membership functions for input (raw scores)
        score['poor'] = fuzz.trapmf(score.universe, [0.0, 0.2, 0.35, 0.45])
        score['fair'] = fuzz.trapmf(score.universe, [0.45, 0.5, 0.6, 0.65])
        score['good'] = fuzz.trapmf(score.universe, [0.65, 0.7, 0.75, 0.8])
        score['very_good'] = fuzz.trapmf(score.universe, [0.8, 0.85, 0.9, 0.95])
        score['excellent'] = fuzz.trapmf(score.universe, [0.9, 0.95, 0.98, 1.0])

        # Define membership functions for category scores 
        category_score['poor'] = fuzz.trapmf(category_score.universe, [0.0, 0.25, 0.33, 0.40])
        category_score['fair'] = fuzz.trapmf(category_score.universe, [0.40, 0.45, 0.53, 0.60])
        category_score['good'] = fuzz.trapmf(category_score.universe, [0.60, 0.65, 0.73, 0.80])
        category_score['very_good'] = fuzz.trapmf(category_score.universe, [0.83, 0.87, 0.89, 0.91])
        category_score['excellent'] = fuzz.trapmf(category_score.universe, [0.89, 0.95, 0.98, 1.0])

        rules = [
            ctrl.Rule(score['poor'], category_score['poor']),
            ctrl.Rule(score['fair'], category_score['fair']),
            ctrl.Rule(score['good'], category_score['good']),
            ctrl.Rule(score['very_good'], category_score['very_good']),
            ctrl.Rule(score['excellent'], category_score['excellent']),
        ]
        
        # Create and return control system
        control_system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystem(rules)    