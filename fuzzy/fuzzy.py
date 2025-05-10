# fuzzy/fuzzy.py

import numpy as np
import skfuzzy.control as ctrl

class FuzzyLogicEvaluator:
    
    def __init__(self):
        self.control_system = None
    
    def _create_control_system(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def evaluate(self, raw_score):
        try:
            if isinstance(raw_score, (int, float)) and not np.isnan(raw_score):
                raw_score = max(0.01, min(raw_score, 0.99))  # Avoid exact boundary values
                # Create a fresh simulation for each evaluation
                simulation = ctrl.ControlSystemSimulation(self.control_system, cache=False)
                simulation.input['score'] = raw_score
                simulation.compute()
                return simulation.output['category_score']
            else:
                return 0.5  # Return a safe middle value
        except Exception as e:
            print(f"[Fuzzy Error] Score {raw_score}: {str(e)}")
            return 0.5  # Return a safe middle value