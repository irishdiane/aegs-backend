# flask-server/scoring/calculator.py

from typing import Dict, List, Any, Optional
import json
import time

class ResultsProcessor:
    
    def __init__(self):
        # Default weights for different evaluation criteria
        self.default_weights = {
            'grammar': 1.0,  # Since we only have grammar for now
            # Add others when implemented
        }
    
    def calculate_weighted_score(self, scores: Dict[str, float], 
                               weights: Optional[Dict[str, float]] = None) -> float:
        if weights is None:
            weights = self.default_weights
        else:
            # Filter weights to only include categories we have scores for
            weights = {k: v for k, v in weights.items() if k in scores}
            
            # If no valid weights, use defaults
            if not weights:
                weights = self.default_weights
            
        total_score = 0.0
        total_weight = sum(weights.values())
        
        for category, weight in weights.items():
            if category in scores:
                total_score += scores[category] * (weight / total_weight)
            
        return round(total_score, 5)
    
    def prepare_result(self, 
                     essay_id: str,
                     raw_scores: Dict[str, float],
                     fuzzy_scores: Dict[str, float],
                     weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        weighted_score = self.calculate_weighted_score(fuzzy_scores, weights)
            
        # Convert scores to percentages for display
        percent_scores = {k: round(v * 100, 1) for k, v in fuzzy_scores.items()}
        
        # Map scores to grade levels
        grade_mapping = {
            (0.9, 1.0): "Excellent (A)",
            (0.8, 0.9): "Very Good (B+)",
            (0.7, 0.8): "Good (B)",
            (0.6, 0.7): "Satisfactory (C+)",
            (0.5, 0.6): "Acceptable (C)",
            (0.0, 0.5): "Needs Improvement (D)"
        }
        
        overall_grade = next((grade for (min_val, max_val), grade in grade_mapping.items() 
                           if min_val <= weighted_score < max_val), "Not Graded")
        
        return {
            "essay_id": essay_id,
            "scores": {
                "raw": raw_scores,
                "fuzzy": fuzzy_scores,
                "percentage": percent_scores
            },
            "weighted_score": round(weighted_score * 100, 1),  # As percentage
            "grade": overall_grade,
            "timestamp": time.time()
        }
        
    def format_for_api(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Create a simplified version for the API response
        api_result = {
            "essay_id": result["essay_id"],
            "scores": result["scores"]["percentage"],
            "weighted_score": result["weighted_score"],
            "grade": result["grade"],
            "score_breakdown": [
                {"category": k, "score": v} 
                for k, v in result["scores"]["percentage"].items()
            ]
        }
        
        return api_result
        
    def batch_process_results(self, 
                            essay_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not essay_results:
            return {"status": "error", "message": "No results to process"}
            
        weighted_scores = [r.get("weighted_score", 0) for r in essay_results]
        
        summary = {
            "total_essays": len(essay_results),
            "avg_score": round(sum(weighted_scores) / len(weighted_scores), 1),
            "min_score": min(weighted_scores),
            "max_score": max(weighted_scores),
            "essays": essay_results
        }
        
        return summary