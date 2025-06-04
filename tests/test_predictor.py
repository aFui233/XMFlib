import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from XMFlib.PairProbML import PairProbPredictor

# Instantiate the predictor
predictor = PairProbPredictor()

# Run prediction with example values
result = predictor.predict(
    facet=100,
    interaction_energy=0.2,
    temperature=400,
    main_coverage=0.5
)

print("Predicted probabilities:", result)