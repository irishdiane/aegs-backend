#fuzzy_graphs.py

import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def plot_membership_functions(fuzzy_input, title="Fuzzy Input Membership Functions"):
    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Loop through all the terms for the fuzzy input (e.g., "poor", "fair", "good")
        for term in fuzzy_input.terms:
            fuzzy_input.view(sim=None, ax=ax)  # Visualize the fuzzy input's terms

        ax.set_title(title)
        
        # Save the figure to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.read()).decode('utf-8')
        
        return img_data
    
    except Exception as e:
        print(f"[ERROR] Error in plotting: {str(e)}")
        return None

