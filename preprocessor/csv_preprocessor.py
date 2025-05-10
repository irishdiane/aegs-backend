# flask-server/preprocessing/csv_processor.py

import pandas as pd
import os
from typing import Dict, Tuple, List, Any

class InputProcessor:
    
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:

        try:
            df = pd.read_csv(file_path)
            # Validate required columns
            required_cols = ['essay_text', 'prompt']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"CSV file is missing required column: {col}")
            
            # Add essay_id if not present
            if 'essay_id' not in df.columns:
                df['essay_id'] = [f"essay_{i}" for i in range(len(df))]
                
            df = df[['essay_id', 'essay_text', 'prompt']]

            return df
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    @staticmethod
    def prepare_for_evaluation(input_data: Dict[str, Any]) -> Tuple[str, str, str]:

        return (
            input_data['essay_id'],
            input_data['essay_text'],
            input_data['prompt']
        )
    
    @staticmethod
    def batch_process_csv(file_path: str) -> List[Dict[str, Any]]:

        df = InputProcessor.read_csv(file_path)
        essays = []
        
        for _, row in df.iterrows():
            essay = {
                'essay_id': row['essay_id'],
                'essay_text': row['essay_text'],
                'prompt': row['prompt']
            }
            essays.append(essay)
            
        return essays