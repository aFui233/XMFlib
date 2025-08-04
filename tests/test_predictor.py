import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from XMFlib.PairProbML import PairProbPredictor

# Instantiate the predictor
predictor = PairProbPredictor()

# Run prediction with example values
result = predictor.predict(
    facet=111,
    interaction_energy=0.3,
    temperature=400,
    main_coverage=0.7
)

print("Predicted probabilities:", result)

# Run prediction for 2NN with example values
result_2nn = predictor.predict_2nn(
    facet=100,
    interaction_energy_1nn=0.18,
    interaction_energy_2nn=0.04,
    temperature=525,
    main_coverage=0.7
)

print("Predicted 2NN probabilities:", result_2nn)