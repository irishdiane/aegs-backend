import language_tool_python
from nltk.tokenize import sent_tokenize

class MechanicsEvaluator:
    
    def __init__(self):
        # Initialize LanguageTool for English
        self.tool = language_tool_python.LanguageTool('en-US')
    
    def evaluate(self, essay_text):
        sentences = sent_tokenize(essay_text)
        
        if not sentences:
            return 0.0  # No text = worst score

        # Issue types to consider
        relevant_types = {'PUNCTUATION', 'TYPOGRAPHY', 'CASING', 'SPELLING'}

        total_relevant_errors = 0
        for sentence in sentences:
            matches = self.tool.check(sentence)
            # Filter for relevant issues only
            filtered_matches = [
                m for m in matches 
                if m.ruleIssueType.upper() in relevant_types
            ]
            total_relevant_errors += len(filtered_matches)

        max_errors = len(sentences) * 7  # 7 errors per sentence as worst case

        raw_score = 1 - (total_relevant_errors / max_errors)

        # Clamp the value between 0 and 1
        final_score = round(min(max(raw_score, 0), 1.0), 5)
        
        return final_score
