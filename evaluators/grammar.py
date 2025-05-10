import language_tool_python
from nltk.tokenize import sent_tokenize

class GrammarEvaluator:
    
    def __init__(self):
        # Initialize LanguageTool for English
        self.tool = language_tool_python.LanguageTool('en-US')
    
    def evaluate(self, essay_text):
        sentences = sent_tokenize(essay_text)
        
        if not sentences:
            return 0.0  # No text = worst score
        
        total_errors = sum(len(self.tool.check(sentence)) for sentence in sentences)
        max_errors = len(sentences) * 7  # 7 errors per sentence as worst case
        
        raw_score = 1 - (total_errors / max_errors)
        
        # Clamp the value between 0 and 1
        final_score = round(min(max(raw_score, 0), 1.0), 5)
        
        return final_score