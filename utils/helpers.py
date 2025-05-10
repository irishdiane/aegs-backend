import os
import pandas as pd
import inspect
from modules.config import get_config
from modules.evaluators import load_all_evaluators
from modules.fuzzy import load_all_fuzzy_evaluators
from modules.processors.input_processor import InputProcessor
from modules.processors.results_processor import ResultsProcessor
from modules.scoring.scale_converter import convert_to_scale
from modules.scoring.rubric import get_rubric_criteria, get_scale_name

class EssayEvaluationEngine:
    def __init__(self):
        # Load evaluators
        self.evaluators = load_all_evaluators()
        
        # Load fuzzy evaluators
        self.fuzzy_evaluators = load_all_fuzzy_evaluators()
        
        # Initialize processors
        self.input_processor = InputProcessor()
        self.results_processor = ResultsProcessor()
        
        # Store configuration
        self.rubric_choice = None
        self.criteria = []
        self.weights = {}
        self.scale_choice = None
    
    def call_evaluator(self, evaluator, method_name, essay_text, prompt):
        """Call evaluator methods dynamically based on their parameter count"""
        method = getattr(evaluator, method_name)
        # Get the number of parameters the method expects
        sig = inspect.signature(method)
        param_count = len(sig.parameters)

        if param_count == 1:  # essay_text only
            return method(essay_text)
        elif param_count >= 2:  # essay_text + prompt
            return method(essay_text, prompt)
        else:
            raise ValueError(f"Unsupported number of parameters: {param_count}")
    
    def configure_system(self):
        """Configure rubric, weights, and scoring scale"""
        # Select rubric type
        print("\n=== Select Rubric Type ===")
        print("1. Rubric 1 (Ideas, Evidence, Structure, Vocabulary)")
        print("2. Rubric 2 (All criteria)")
        print("3. Rubric 3 (Ideas, Grammar, Language, Vocabulary)")
        
        self.rubric_choice = ""
        while self.rubric_choice not in ["1", "2", "3"]:
            self.rubric_choice = input("Enter your choice (1/2/3): ")
        
        # Set criteria based on rubric choice
        self.criteria = get_rubric_criteria(self.rubric_choice)
        
        # Configure weights
        self.weights = {}
        print("\n=== Configure Weights (Total must be 100%) ===")
        
        while True:
            total_weight = 0
            for criterion in self.criteria:
                display_name = criterion.replace("_", " ").capitalize()
                while True:
                    try:
                        weight = int(input(f"Enter weight for {display_name} (%): "))
                        if weight < 0:
                            print("Weight cannot be negative. Please try again.")
                        else:
                            break
                    except ValueError:
                        print("Please enter a valid integer.")
                
                self.weights[criterion] = weight
                total_weight += weight
            
            if total_weight == 100:
                break
            else:
                print(f"Total weight must be 100%. Current total: {total_weight}%. Please try again.")
        
        # Select scoring scale
        print("\n=== Select Scoring Scale ===")
        print("1. 5-point scale (1-5)")
        print("2. 20-point scale (1-20)")
        print("3. Letter grades (A-E)")
        print("4. Letter grades with plus/minus (A+, A, A-, etc.)")
        print("5. 100-point scale (0-100)")
        print("6. 50-point scale (0-50)")
        
        self.scale_choice = ""
        while self.scale_choice not in ["1", "2", "3", "4", "5", "6"]:
            self.scale_choice = input("Enter your choice (1-6): ")
    
    def process_csv_file(self, file_path):
        """Process a CSV file containing essays and prompts"""
        print(f"Processing CSV file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return None
        
        try:
            # Configure the system first
            self.configure_system()
            
            # Directly read the CSV file using pandas
            data = pd.read_csv(file_path)
            
            # Process the data if needed using input processor methods
            if hasattr(self.input_processor, 'process_data'):
                data = self.input_processor.process_data(data)
            elif hasattr(self.input_processor, 'process'):
                data = self.input_processor.process(data)
            
            # Check if data is empty
            if len(data) == 0:
                print("Error: CSV file is empty")
                return None
            
            # Check if required columns exist
            if 'essay_text' not in data.columns or 'prompt' not in data.columns:
                print("Error: CSV must have 'essay_text' and 'prompt' columns")
                return None
            
            print(f"Found {len(data)} essays to evaluate.")
            
            # Create a new column for scores
            data['final_score'] = None
            
            # Process each essay
            for idx, row in data.iterrows():
                essay_text = row['essay_text']
                prompt = row['prompt']
                
                print(f"Evaluating essay {idx+1}/{len(data)}...")
                
                # Evaluate essay and get final score
                evaluation_result = self.evaluate_essay(essay_text, prompt)
                
                # Filter results based on selected criteria
                filtered_results = {}
                for criterion in self.criteria:
                    if criterion in evaluation_result:
                        filtered_results[criterion] = evaluation_result[criterion]
                        filtered_results[f"fuzzy_{criterion}"] = evaluation_result.get(f"fuzzy_{criterion}", 0.01)
                
                # Calculate fuzzy weighted score
                fuzzy_weighted_score = 0
                for criterion in self.criteria:
                    fuzzy_key = f"fuzzy_{criterion}"
                    if fuzzy_key in filtered_results:
                        fuzzy_weighted_score += filtered_results[fuzzy_key] * (self.weights[criterion] / 100)
                
                # Convert to selected scale
                final_score = convert_to_scale(fuzzy_weighted_score, self.scale_choice)
                
                # Add score to dataframe
                data.at[idx, 'final_score'] = final_score
            
            # Save results to new CSV file
            output_path = os.path.splitext(file_path)[0] + "_scored.csv"
            data.to_csv(output_path, index=False)
            
            print(f"\nEvaluation complete! Results saved to: {output_path}")
            
            return {
                "input_file": file_path,
                "output_file": output_path,
                "essays_evaluated": len(data),
                "rubric_type": self.rubric_choice,
                "criteria": self.criteria,
                "weights": self.weights,
                "scale_type": self.scale_choice
            }
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_manual_input(self, essay_text, prompt):
        """Process manually entered essay and prompt"""
        print("Processing manual input...")
        
        if not essay_text or not prompt:
            print("Error: Essay and prompt cannot be empty")
            return None
        
        try:
            # Configure the system first
            self.configure_system()
            
            print("Evaluating essay...")
            
            # Evaluate essay
            evaluation_result = self.evaluate_essay(essay_text, prompt)
            
            # Filter results based on selected criteria
            filtered_results = {}
            for criterion in self.criteria:
                if criterion in evaluation_result:
                    filtered_results[criterion] = evaluation_result[criterion]
                    filtered_results[f"fuzzy_{criterion}"] = evaluation_result.get(f"fuzzy_{criterion}", 0.01)
            
            # Calculate weighted score based on selected criteria and weights
            weighted_score = 0
            for criterion in self.criteria:
                if criterion in filtered_results:
                    weighted_score += filtered_results[criterion] * (self.weights[criterion] / 100)
            
            # Calculate fuzzy weighted score
            fuzzy_weighted_score = 0
            for criterion in self.criteria:
                fuzzy_key = f"fuzzy_{criterion}"
                if fuzzy_key in filtered_results:
                    fuzzy_weighted_score += filtered_results[fuzzy_key] * (self.weights[criterion] / 100)
            
            # Convert to selected scale
            final_score = convert_to_scale(fuzzy_weighted_score, self.scale_choice)
            
            # Display results
            print("\n=== Evaluation Results ===")
            print(f"Rubric Type: {self.rubric_choice}")
            
            print("\nIndividual Criteria Scores:")
            for criterion in self.criteria:
                display_name = criterion.replace("_", " ").capitalize()
                if criterion in filtered_results:
                    raw_score = filtered_results[criterion]
                    fuzzy_score = filtered_results.get(f"fuzzy_{criterion}", 0.01)
                    weight = self.weights[criterion]
                    print(f"- {display_name}: Raw={raw_score:.2f}, Fuzzy={fuzzy_score:.2f}, Weight={weight}%")
            
            print(f"\nWeighted Score: {weighted_score:.2f}")
            print(f"Fuzzy Weighted Score: {fuzzy_weighted_score:.2f}")
            print(f"Final Score ({get_scale_name(self.scale_choice)}): {final_score}")
            
            return {
                "rubric_type": self.rubric_choice,
                "criteria": self.criteria,
                "weights": self.weights,
                "raw_scores": {k: filtered_results[k] for k in self.criteria if k in filtered_results},
                "fuzzy_scores": {f"fuzzy_{k}": filtered_results.get(f"fuzzy_{k}", 0.01) for k in self.criteria},
                "weighted_score": weighted_score,
                "fuzzy_weighted_score": fuzzy_weighted_score,
                "scale_type": self.scale_choice,
                "final_score": final_score
            }
            
        except Exception as e:
            print(f"Error processing manual input: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_essay(self, essay_text, prompt):
        """Evaluate an essay using all criteria and fuzzify results"""
        try:
            # Run individual evaluators
            raw_scores = {}
            
            # Process each criterion
            for criterion, evaluator in self.evaluators.items():
                raw_scores[criterion] = self.call_evaluator(evaluator, 'evaluate', essay_text, prompt)
            
            # Fuzzify scores with error handling
            fuzzy_scores = {}
            
            # Helper function for safe fuzzification
            def safe_fuzzify(evaluator, score, criterion_name):
                try:
                    # Ensure score is within valid range
                    if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                        print(f"Warning: Invalid {criterion_name} score {score}, using default 0.5")
                        return 0.5
                    
                    # Create fresh simulation for each evaluation
                    result = evaluator.evaluate(score)
                    
                    # Check if result is valid
                    if result is None or not isinstance(result, (int, float)) or result < 0 or result > 1:
                        print(f"Warning: Invalid fuzzy result for {criterion_name}: {result}, using default 0.5")
                        return 0.5
                    
                    return result
                except Exception as e:
                    print(f"[Fuzzy Error] {criterion_name} Score {score}: {str(e)}")
                    return 0.5
            
            # Safely fuzzify each score
            for criterion, score in raw_scores.items():
                fuzzy_evaluator = self.fuzzy_evaluators.get(criterion)
                if fuzzy_evaluator:
                    display_name = criterion.replace("_", " ").capitalize()
                    fuzzy_scores[f"fuzzy_{criterion}"] = safe_fuzzify(fuzzy_evaluator, score, display_name)
            
            # Calculate overall score using existing weights in results processor
            overall_score = self.results_processor.calculate_weighted_score(raw_scores)
            fuzzy_overall = self.results_processor.calculate_weighted_score(fuzzy_scores)
            
            # Combine all results
            result = {
                **raw_scores,
                **fuzzy_scores,
                'overall_score': overall_score,
                'fuzzy_overall': fuzzy_overall
            }
            
            return result
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Return safe default values
            default_result = {
                'error': str(e),
                'overall_score': 0.5,
                'fuzzy_overall': 0.5
            }
            
            # Add default scores for all known criteria
            for criterion in self.evaluators.keys():
                default_result[criterion] = 0.5
                default_result[f"fuzzy_{criterion}"] = 0.5
                
            return default_result