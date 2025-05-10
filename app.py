#app.py

from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import os
import json
from werkzeug.utils import secure_filename
import tempfile
import inspect
import traceback
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
from main_server import EssayEvaluationSystem
from visualization.fuzzy_graphs import plot_membership_functions
from preprocessor.csv_preprocessor import InputProcessor

app = Flask(__name__)

# Configure file upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/evaluate/text', methods=['POST'])
def evaluate_text():
    print("Reached /grade route")
    """Endpoint for evaluating a single essay"""
    try:
        # Parse request data
        data = request.json
        
        if not data or 'essay_text' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing essay_text or prompt'}), 400
        
        essay_text = data['essay_text']
        prompt = data['prompt']
        
        # Create evaluation system
        system = EssayEvaluationSystem()
        
        # Configure system programmatically based on request
        if 'rubric_choice' in data:
            system.rubric_choice = str(data['rubric_choice'])
        else:
            system.rubric_choice = "2"  # Default to all criteria
            
        # Set criteria based on rubric choice
        if system.rubric_choice == "1":
            system.criteria = ["ideas", "evidence", "organization", "language_tone"]
        elif system.rubric_choice == "2":
            system.criteria = ["ideas", "evidence", "language_tone", "grammar"]
        else:  # rubric_choice == "3"
            system.criteria = ["ideas", "evidence", "organization", "language_tone", "grammar", "mechanics", "vocabulary"]
        
        # Set weights

        if 'weights' in data and data['weights']:
            # Convert from frontend weights to backend weights format
            raw_weights = data['weights']
            total = sum(raw_weights.values())
            
            # Normalize weights to sum up to 100%
            system.weights = {k: (v / total) * 100 for k, v in raw_weights.items() if k in system.criteria}
        else:
            # Equal weights
            weight_value = 100 / len(system.criteria)
            system.weights = {criterion: weight_value for criterion in system.criteria}
            
        # Set scale choice
        if 'scale_choice' in data:
            system.scale_choice = str(data['scale_choice'])
        else:
            system.scale_choice = "5"  # Default to 100-point scale
        
        # Evaluate essay
        result = system.evaluate_essay(essay_text, prompt)
        
        # Format response according to what the frontend expects
        response = {
            'overall_score': result.get('fuzzy_overall', 0),
            'criteria_scores': {criterion: result.get(f'fuzzy_{criterion}', 0) for criterion in system.criteria},
            'scale_type': system.scale_choice,
            'final_score': system.convert_to_scale(result.get('fuzzy_overall', 0), system.scale_choice)
        }
        
        return jsonify(response)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400
        
        weights_json = request.form.get('weights', '{}')
        print(f"DEBUG - Received weights_json: {weights_json}")
        try:
            weights = json.loads(weights_json)
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON decode error: {e}")
            weights = {}

        rubric_choice = request.form.get('rubric_choice', '2')
        scale_choice = request.form.get('scale_choice', '5')
        print(f"DEBUG - Received rubric_choice: {rubric_choice}, scale_choice: {scale_choice}")

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create evaluation system
        system = EssayEvaluationSystem()
                    
        # Set configuration directly based on frontend inputs
        system.rubric_choice = str(rubric_choice)
        
        # Set criteria based on rubric choice
        if system.rubric_choice == "1":
            system.criteria = ["ideas", "evidence", "organization", "language_tone"]
        elif system.rubric_choice == "2":
            system.criteria = ["ideas", "evidence", "language_tone", "grammar"]
        else:  # rubric_choice == "3"
            system.criteria = ["ideas", "evidence", "organization", "language_tone", "grammar", "mechanics", "vocabulary"]
        
        # Convert from frontend weights to backend weights format
        if weights:
            total = sum(weights.values())
            # Set weights from frontend
            system.weights = {k: (v / total) * 100 for k, v in weights.items() if k in system.criteria}
        else:
            # Equal weights
            weight_value = 100 / len(system.criteria)
            system.weights = {criterion: weight_value for criterion in system.criteria}
            
        system.scale_choice = str(scale_choice)

        # Process the CSV file using the InputProcessor
        essays = InputProcessor.batch_process_csv(filepath)

        # Evaluate each essay and store the results
        result = system.process_csv_file(filepath)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # The output file is already created by process_csv_file
        output_filepath = result["output_file"]
        output_filename = os.path.basename(output_filepath)

        # Return the file for download
        return send_file(output_filepath, 
                        mimetype='text/csv',
                        download_name=output_filename,
                        as_attachment=True)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/convert-score', methods=['POST'])
def convert_score():
    """Endpoint for converting a fuzzy score to different scale formats"""
    try:
        data = request.json
        
        if not data or 'score' not in data:
            return jsonify({'error': 'Missing score parameter'}), 400
            
        score = float(data['score'])
        scale = data.get('scale', '5')  # Get scale from request, default to 5
        
        # Create evaluation system just for conversion
        system = EssayEvaluationSystem()
        
        # Convert score using the requested scale
        converted = system.convert_to_scale(score, scale)
        
        return jsonify({
            'original': score,
            'converted': converted,
            'scale_type': system.get_scale_name(scale)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/fuzzy-graph/<criterion>', methods=['GET'])
def fuzzy_graph(criterion):
    try:
        from fuzzy.criteria_fuzzy import f_grammar, f_ideas, f_structure, f_evidence, f_language_tone, f_vocab, f_mp
        
        # Dynamically map criterion to fuzzy evaluator class
        mapping = {
            "grammar": f_grammar.GrammarFuzzyEvaluator,
            "ideas": f_ideas.IdeasFuzzyEvaluator,
            "organization": f_structure.OrgFuzzyEvaluator,
            "evidence": f_evidence.EvidenceFuzzyEvaluator,
            "language_tone": f_language_tone.LangFuzzyEvaluator,
            "vocabulary": f_vocab.VocabFuzzyEvaluator,
            "mechanics": f_mp.MechanicsFuzzyEvaluator
        }
        
        if criterion not in mapping:
            return jsonify({"error": "Invalid criterion"}), 400
        
        # Create a new instance of the evaluator
        evaluator_class = mapping[criterion]
        evaluator = evaluator_class()
        
        # We need to access the code that defines the membership functions        
        # Create a temporary evaluator to extract membership functions
        # This ensures we get a fresh implementation with all variables defined
        temp_evaluator = evaluator_class()
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract and recreate the score antecedent based on the pattern in your example
        # The score is on a 0 to 1 scale with 0.01 increments
        score_universe = np.arange(0, 1.1, 0.01)
        
        # Extract the rules from the control system
        # Since we can't directly access the variables, we'll recreate the membership functions
        # by inspecting how the _create_control_system method is implemented
        
        # Get the source code of the _create_control_system method
        import inspect
        source = inspect.getsource(temp_evaluator._create_control_system)
        
        # Parse the membership functions from the source code
        score_mfs = {}
        category_mfs = {}
        
        import re
        
        # Find and extract score membership functions
        score_pattern = re.compile(r"score\['([^']+)'\]\s*=\s*fuzz\.trapmf\(score\.universe,\s*\[([\d\., ]+)\]\)")
        for match in score_pattern.finditer(source):
            term_name = match.group(1)
            points = [float(p.strip()) for p in match.group(2).split(',')]
            score_mfs[term_name] = points
        
        # If we found membership functions, plot them
        if score_mfs:
            for term_name, points in score_mfs.items():
                membership = fuzz.trapmf(score_universe, points)
                ax.plot(score_universe, membership, label=f"{term_name}")
            
            ax.set_title(f"Membership Functions for {criterion.capitalize()}")
            ax.legend(loc='best')
            ax.grid(True)
            ax.set_xlabel('Score')
            ax.set_ylabel('Membership Degree')
            
            # Save the figure to a BytesIO object
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)  # Close the figure to prevent memory leaks
            output.seek(0)
            
            # Return the image
            return send_file(output, mimetype='image/png')
        
        # If we couldn't find membership functions through parsing,
        # try a different approach
        # Recreate the fuzzy logic system and try to access the variables directly
        evaluator._create_control_system()
        
        # Create a direct approach that works with scikit-fuzzy's structure
        temp_evaluator = evaluator_class()
        
        # Access the control system rules directly
        control_system = temp_evaluator._create_control_system()
        
        # In scikit-fuzzy, the rules contain references to the original variables
        if hasattr(control_system, 'rules'):
            # Extract the first rule and get the antecedent
            if control_system.rules:
                # Get the first rule
                first_rule = list(control_system.rules)[0]
                
                # Extract the antecedent variable (the score)
                if hasattr(first_rule, 'antecedent'):
                    for var_name, var in first_rule.antecedent.terms.items():
                        if hasattr(var, 'parent'):
                            input_var = var.parent
                            
                            # Plot each term's membership function
                            for term_name, term in input_var.terms.items():
                                ax.plot(input_var.universe, term.mf, label=term_name)
                            
                            ax.set_title(f"Membership Functions for {criterion.capitalize()}")
                            ax.legend(loc='best')
                            ax.grid(True)
                            ax.set_xlabel('Score')
                            ax.set_ylabel('Membership Degree')
                            
                            # Save the figure to a BytesIO object
                            output = io.BytesIO()
                            FigureCanvas(fig).print_png(output)
                            plt.close(fig)  # Close the figure to prevent memory leaks
                            output.seek(0)
                            
                            # Return the image
                            return send_file(output, mimetype='image/png')
                        
        #WORST CASE SCENARIO

        # If we reach here, we couldn't access the variables through any method
        # Last resort: manually create the membership functions
        
        default_score_mfs = {
            'poor': [0.0, 0.4, 0.43, 0.45],
            'fair': [0.45, 0.48, 0.50, 0.53],
            'good': [0.53, 0.55, 0.58, 0.63],
            'very_good': [0.63, 0.73, 0.78, 0.83],
            'excellent': [0.83, 0.88, 0.93, 1.0]
        }
        
        score_universe = np.arange(0, 1.1, 0.01)
        
        for term_name, points in default_score_mfs.items():
            membership = fuzz.trapmf(score_universe, points)
            ax.plot(score_universe, membership, label=term_name)
        
        ax.set_title(f"Membership Functions for {criterion.capitalize()} (Default)")
        ax.legend(loc='best')
        ax.grid(True)
        ax.set_xlabel('Score')
        ax.set_ylabel('Membership Degree')
        
        # Save the figure to a BytesIO object
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)  # Close the figure to prevent memory leaks
        output.seek(0)
        
        # Return the image
        return send_file(output, mimetype='image/png')

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception occurred: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return send_file('../build/index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_file(f'../build/{path}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)