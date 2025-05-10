#main_server.py

import os
import pandas as pd
import inspect
import json
import traceback
from preprocessor.word2vec_singleton import get_word2vec_model

try:
    from evaluators.grammar import GrammarEvaluator
    from evaluators.ideas import IdeasEvaluator
    from evaluators.stucture import OrganizationEvaluator
    from evaluators.evidence import EvidenceEvaluator
    from evaluators.language_tone import LanguageToneEvaluator
    from evaluators.vocab import VocabularyEvaluator
    from evaluators.mp import MechanicsEvaluator
    # Import fuzzy evaluators
    from fuzzy.criteria_fuzzy.f_grammar import GrammarFuzzyEvaluator
    from fuzzy.criteria_fuzzy.f_ideas import IdeasFuzzyEvaluator
    from fuzzy.criteria_fuzzy.f_structure import OrgFuzzyEvaluator
    from fuzzy.criteria_fuzzy.f_evidence import EvidenceFuzzyEvaluator
    from fuzzy.criteria_fuzzy.f_language_tone import LangFuzzyEvaluator 
    from fuzzy.criteria_fuzzy.f_vocab import VocabFuzzyEvaluator
    from fuzzy.criteria_fuzzy.f_mp import MechanicsFuzzyEvaluator

    # Import processors
    from preprocessor.csv_preprocessor import InputProcessor
    from scoring.calculator import ResultsProcessor

except ImportError:
    # If imports fail, create mock classes for testing
    print("Warning: Using mock evaluators as imports failed. Check your project structure.")
    
    class MockEvaluator:
        def evaluate(self, *args):
            return 0.75  # Default score
    
    # Create mock instances
    GrammarEvaluator = type('GrammarEvaluator', (MockEvaluator,), {})
    IdeasEvaluator = type('IdeasEvaluator', (MockEvaluator,), {})
    OrganizationEvaluator = type('OrganizationEvaluator', (MockEvaluator,), {})
    EvidenceEvaluator = type('EvidenceEvaluator', (MockEvaluator,), {})
    LanguageToneEvaluator = type('LanguageToneEvaluator', (MockEvaluator,), {})
    VocabularyEvaluator = type('VocabularyEvaluator', (MockEvaluator,), {})
    MechanicsEvaluator = type('MechanicsEvaluator', (MockEvaluator,), {})   

    GrammarFuzzyEvaluator = type('GrammarFuzzyEvaluator', (MockEvaluator,), {})
    IdeasFuzzyEvaluator = type('IdeasFuzzyEvaluator', (MockEvaluator,), {})
    OrgFuzzyEvaluator = type('OrgFuzzyEvaluator', (MockEvaluator,), {})
    EvidenceFuzzyEvaluator = type('EvidenceFuzzyEvaluator', (MockEvaluator,), {})
    LangFuzzyEvaluator = type('LangFuzzyEvaluator', (MockEvaluator,), {})
    VocabFuzzyEvaluator = type('VocabFuzzyEvaluator', (MockEvaluator,), {})
    MechanicsFuzzyEvaluator = type('MechanicsFuzzyEvaluator', (MockEvaluator,), {})   

    # Mock processors
    class InputProcessor:
        def process_data(self, data):
            return data
        
        def process(self, data):
            return data
    
    class ResultsProcessor:
        def calculate_weighted_score(self, scores):
            return 0.75

class EssayEvaluationSystem:
    def __init__(self):
        # Get shared Word2Vec model instance
        self.word2vec_model = get_word2vec_model()
        
        # Initialize evaluators with the shared model
        self.grammar_evaluator = GrammarEvaluator()
        self.ideas_evaluator = IdeasEvaluator(word2vec_model=self.word2vec_model)
        self.organization_evaluator = OrganizationEvaluator(word2vec_model=self.word2vec_model)
        self.evidence_evaluator = EvidenceEvaluator(word2vec_model=self.word2vec_model)
        self.language_tone_evaluator = LanguageToneEvaluator(word2vec_model=self.word2vec_model)
        self.vocabulary_evaluator = VocabularyEvaluator(word2vec_model=self.word2vec_model)
        self.mechanics_evaluator = MechanicsEvaluator()

        # Initialize fuzzy evaluators
        self.grammar_fuzzy = GrammarFuzzyEvaluator()
        self.ideas_fuzzy = IdeasFuzzyEvaluator()
        self.organization_fuzzy = OrgFuzzyEvaluator()
        self.evidence_fuzzy = EvidenceFuzzyEvaluator()
        self.language_tone_fuzzy = LangFuzzyEvaluator()
        self.vocabulary_fuzzy = VocabFuzzyEvaluator()
        self.mechanics_fuzzy = MechanicsFuzzyEvaluator()
        
        # Initialize processors
        self.input_processor = InputProcessor()
        self.results_processor = ResultsProcessor()
        # Default configuration
        self.rubric_choice = None
        self.criteria = []
        self.weights = {}
        self.scale_choice = "5"  # Default to 100-point scale

    def call_evaluator(self, evaluator, method_name, essay_text, prompt):
        method = getattr(evaluator, method_name)
        
        try:
            # Get the number of parameters the method expects
            sig = inspect.signature(method)
            param_count = len(sig.parameters)

            if param_count == 1:  # essay_text only
                return method(essay_text)
            elif param_count >= 2:  # essay_text + prompt
                return method(essay_text, prompt)
            else:
                return 0.5  # Default if method signature is unexpected
        except Exception:
            # Fallback if inspect fails
            try:
                return method(essay_text, prompt)
            except:
                try:
                    return method(essay_text)
                except:
                    return 0.5  # Default score
        
    def get_scale_name(self, scale_choice):
        scale_names = {
            "1": "5-point scale",
            "2": "20-point scale",
            "3": "Letter grades (A-E)",
            "4": "Letter grades with plus/minus",
            "5": "100-point scale",
            "6": "50-point scale"
        }
        return scale_names.get(scale_choice, "Unknown scale")
    
    def convert_to_scale(self, fuzzy_score, scale_choice):
        # Ensure fuzzy_score is within [0, 1] range
        fuzzy_score = max(0, min(fuzzy_score, 1))
        
        if scale_choice == "1":  # 5-point scale
            raw = 1 + fuzzy_score * 4
            return f"{raw:.1f}/5"
            
        elif scale_choice == "2":  # 20-point scale
            raw = 1 + fuzzy_score * 19
            return f"{raw:.1f}/20"
            
        elif scale_choice == "3":  # Letter grades A-E
            if fuzzy_score >= 0.9:
                return "A"
            elif fuzzy_score >= 0.75:
                return "B"
            elif fuzzy_score >= 0.6:
                return "C"
            elif fuzzy_score >= 0.4:
                return "D"
            else:
                return "E"
                
        elif scale_choice == "4":  # Letter grades with plus/minus
            if fuzzy_score >= 0.97:
                return "A+"
            elif fuzzy_score >= 0.93:
                return "A"
            elif fuzzy_score >= 0.9:
                return "A-"
            elif fuzzy_score >= 0.87:
                return "B+"
            elif fuzzy_score >= 0.83:
                return "B"
            elif fuzzy_score >= 0.8:
                return "B-"
            elif fuzzy_score >= 0.77:
                return "C+"
            elif fuzzy_score >= 0.73:
                return "C"
            elif fuzzy_score >= 0.7:
                return "C-"
            elif fuzzy_score >= 0.67:
                return "D+"
            elif fuzzy_score >= 0.63:
                return "D"
            elif fuzzy_score >= 0.6:
                return "D-"
            else:
                return "F"
                
        elif scale_choice == "5":  # 100-point scale
            raw = fuzzy_score * 100
            return f"{raw:.1f}/100"
            
        elif scale_choice == "6":  # 50-point scale
            raw = fuzzy_score * 50
            return f"{raw:.1f}/50"
            
        else:
            return f"{fuzzy_score:.2f}"  # Default to raw fuzzy score
    
    def evaluate_essay(self, essay_text, prompt):
        """Evaluate an essay using all criteria and fuzzify results"""
        try:
            # Run individual evaluators - dynamically determine arguments
            grammar_score = self.call_evaluator(self.grammar_evaluator, 'evaluate', essay_text, prompt)
            ideas_score = self.call_evaluator(self.ideas_evaluator, 'evaluate', essay_text, prompt)
            organization_score = self.call_evaluator(self.organization_evaluator, 'evaluate', essay_text, prompt)
            evidence_score = self.call_evaluator(self.evidence_evaluator, 'evaluate', essay_text, prompt)
            language_tone_score = self.call_evaluator(self.language_tone_evaluator, 'evaluate', essay_text, prompt)
            vocabulary_score = self.call_evaluator(self.vocabulary_evaluator, 'evaluate', essay_text, prompt)
            mechanics_score = self.call_evaluator(self.mechanics_evaluator, 'evaluate', essay_text, prompt)

            # Create dictionary of raw scores
            raw_scores = {
                'grammar': grammar_score,
                'ideas': ideas_score,
                'organization': organization_score,
                'evidence': evidence_score,
                'language_tone': language_tone_score,
                'vocabulary': vocabulary_score,
                'mechanics': mechanics_score
            }
            
            # Fuzzify scores with error handling
            fuzzy_scores = {}
            
            # Helper function for safe fuzzification
            def safe_fuzzify(evaluator, score, criterion_name):
                try:
                    # Ensure score is within valid range
                    if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                        return 0.5
                    
                    # Create fresh simulation for each evaluation
                    result = evaluator.evaluate(score)
                    
                    # Check if result is valid
                    if result is None or not isinstance(result, (int, float)) or result < 0 or result > 1:
                        return 0.5
                    
                    return result
                except Exception as e:
                    return 0.5
            
            # Safely fuzzify each score
            fuzzy_grammar = safe_fuzzify(self.grammar_fuzzy, grammar_score, "Grammar")
            fuzzy_ideas = safe_fuzzify(self.ideas_fuzzy, ideas_score, "Ideas")
            fuzzy_organization = safe_fuzzify(self.organization_fuzzy, organization_score, "Organization")
            fuzzy_evidence = safe_fuzzify(self.evidence_fuzzy, evidence_score, "Evidence")
            fuzzy_language_tone = safe_fuzzify(self.language_tone_fuzzy, language_tone_score, "Language Tone")
            fuzzy_vocabulary = safe_fuzzify(self.vocabulary_fuzzy, vocabulary_score, "Vocabulary")
            fuzzy_mechanics = safe_fuzzify(self.mechanics_fuzzy, mechanics_score, "Mechanics")

            # Create dictionary of fuzzy scores
            fuzzy_scores = {
                'fuzzy_grammar': fuzzy_grammar,
                'fuzzy_ideas': fuzzy_ideas,
                'fuzzy_organization': fuzzy_organization,
                'fuzzy_evidence': fuzzy_evidence,
                'fuzzy_language_tone': fuzzy_language_tone,
                'fuzzy_vocabulary': fuzzy_vocabulary,
                'fuzzy_mechanics': fuzzy_mechanics
            }
            
            # If we have criteria and weights, override the overall fuzzy score
            if self.criteria and self.weights:
                fuzzy_weighted_score = 0
                for criterion in self.criteria:
                    fuzzy_key = f"fuzzy_{criterion}"
                    if fuzzy_key in fuzzy_scores and criterion in self.weights:
                        fuzzy_weighted_score += fuzzy_scores[fuzzy_key] * (self.weights[criterion] / 100)
                fuzzy_overall = fuzzy_weighted_score
            
            # Combine all results
            result = {
                **raw_scores,
                **fuzzy_scores
            }
            
            return result
            
        except Exception as e:
            # Return safe default values
            print(f"Error evaluating essay: {str(e)}")
            traceback.print_exc()
            return {
                'error': str(e),
                'grammar': 0.5,
                'ideas': 0.5,
                'organization': 0.5,
                'evidence': 0.5,
                'language_tone': 0.5,
                'vocabulary': 0.5,
                'mechanics': 0.5,
                'fuzzy_grammar': 0.5,
                'fuzzy_ideas': 0.5, 
                'fuzzy_organization': 0.5,
                'fuzzy_evidence': 0.5,
                'fuzzy_language_tone': 0.5,
                'fuzzy_vocabulary': 0.5,
                'fuzzy_mechanics': 0.5,
            }
            
    def process_csv_file(self, file_path):
        print("reached")
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File {file_path} not found"}

        try:
            # Use the InputProcessor to read and validate the CSV
            essays = self.input_processor.batch_process_csv(file_path)

            if not essays:
                return {"error": "CSV file is empty or invalid"}

            # Prepare an output dataframe
            results = []

            for essay in essays:
                essay_id, essay_text, prompt = self.input_processor.prepare_for_evaluation(essay)

                # Evaluate the essay
                evaluation_result = self.evaluate_essay(essay_text, prompt)

                # Calculate fuzzy weighted score - only using selected criteria
                fuzzy_weighted_score = 0
                for criterion in self.criteria:
                    fuzzy_key = f"fuzzy_{criterion}"
                    if fuzzy_key in evaluation_result and criterion in self.weights:
                        fuzzy_weighted_score += evaluation_result.get(fuzzy_key, 0) * (self.weights[criterion] / 100)

                # Start with basic essay information
                output_row = {
                    "essay_id": essay_id,
                    "essay_text": essay_text,
                    "prompt": prompt,
                }
                
                # Add only the criteria from the selected rubric
                for criterion in self.criteria:
                    fuzzy_key = f"fuzzy_{criterion}"
                    output_row[fuzzy_key] = evaluation_result.get(fuzzy_key, 0)

                # Add multiple scaled scores
                scale_names = {
                    "1": "5_point_score",
                    "2": "20_point_score",
                    "3": "letter_grade",
                    "4": "letter_grade_pm",
                    "5": "100_point_score",
                    "6": "50_point_score"
                }

                for scale_id, col_name in scale_names.items():
                    output_row[col_name] = self.convert_to_scale(fuzzy_weighted_score, scale_id)

                results.append(output_row)

            # Convert results to DataFrame and save
            df_output = pd.DataFrame(results)
            output_path = os.path.splitext(file_path)[0] + "_scored.csv"
            df_output.to_csv(output_path, index=False)

            return {
                "input_file": file_path,
                "output_file": output_path,
                "essays_evaluated": len(df_output),
                "rubric_type": self.rubric_choice,
                "criteria": self.criteria,
                "weights": self.weights,
                "scale_type": self.scale_choice
            }

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error processing CSV file: {str(e)}"}