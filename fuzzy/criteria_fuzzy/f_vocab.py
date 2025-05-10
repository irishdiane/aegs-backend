import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from fuzzy.fuzzy import FuzzyLogicEvaluator

class VocabFuzzyEvaluator(FuzzyLogicEvaluator):
    
    def __init__(self):
        """Initialize the evidence fuzzy logic system."""
        super().__init__()
        self.control_system = self._create_control_system()
    
    def _create_control_system(self):
        # Define fuzzy variables
        score = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'score')
        category_score = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'category_score')
        
        # Define membership functions for input (raw scores)
        score['poor'] = fuzz.trapmf(score.universe, [0.0, 0.1, 0.15, 0.20])
        score['fair'] = fuzz.trapmf(score.universe, [0.20, 0.22, 0.24, 0.25])
        score['good'] = fuzz.trapmf(score.universe, [0.25, 0.30, 0.40, 0.45])
        score['very_good'] = fuzz.trapmf(score.universe, [0.45, 0.50, 0.55, 0.60])
        score['excellent'] = fuzz.trapmf(score.universe, [0.60, 0.75, 0.85, 1.0])

        # Define membership functions for category scores
        category_score['poor'] = fuzz.trapmf(category_score.universe, [0.0, 0.15, 0.23, 0.30])
        category_score['fair'] = fuzz.trapmf(category_score.universe, [0.30, 0.35, 0.43, 0.50])
        category_score['good'] = fuzz.trapmf(category_score.universe, [0.50, 0.60, 0.65, 0.75])
        category_score['very_good'] = fuzz.trapmf(category_score.universe, [0.73, 0.75, 0.78, 0.80])
        category_score['excellent'] = fuzz.trapmf(category_score.universe, [0.80, 0.85, 0.93, 1.0])

        # Define fuzzy rules
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