import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from XMFlib.PairProbML import PairProbPredictor
from XMFlib.CovML import CovPredictor

# Instantiate the predictor
pair_prob_predictor = PairProbPredictor()
cov_predictor = CovPredictor()

# Run prediction with example values
result = pair_prob_predictor.predict(
    facet=111,
    interaction_energy=0.3,
    temperature=400,
    main_coverage=0.7
)

print("Predicted probabilities:", result)

# Run prediction for 2NN with example values
result_2nn = pair_prob_predictor.predict_2nn(
    facet=100,
    interaction_energy_1nn=0.18,
    interaction_energy_2nn=0.04,
    temperature=525,
    main_coverage=0.7
)

print("Predicted 2NN probabilities:", result_2nn)

# Run coverage prediction with example values
cov_result = cov_predictor.predict(
    facet=111,
    interaction_energy=0.18,
    adsorption_energy=-0.86,
    temperature=400
)

print("Predicted coverage:", cov_result)