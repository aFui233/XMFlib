import os
import torch
from .io import load_model
from .utils import eps_kbt
from .models import MLP

class PairProbPredictor:
    SUPPORTED_FACETS = ['100', '111']
    OUTPUT_NAMES = ['vacancy_pair', 'species_pair', 'species_vacancy_pair']

    def __init__(self, model_dir=None):
        # Set model directory; use default if not provided
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_dir = model_dir

    def predict(self, facet, interaction_energy, temperature, main_coverage,
                model_type='mlp', task='a2p'):
        # Ensure facet is string
        facet = str(facet)
        if facet not in self.SUPPORTED_FACETS:
            raise ValueError(f"Facet '{facet}' not supported. Supported: {self.SUPPORTED_FACETS}")

        # Build the model file name and path
        model_file = f'{model_type}_{task}_{facet}.pth'
        model_path = os.path.join(self.model_dir, model_file)
        
        # Load the trained model
        model = load_model(model_path, model_class=MLP)
        
        # Convert interaction energy to dimensionless form
        dimless_eps = eps_kbt(interaction_energy, temperature)
        
        # Prepare input features (feature order must match training)
        X = [[dimless_eps, main_coverage]]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            y_pred = model(X_tensor).numpy()

        # Return predictions as a dictionary
        return dict(zip(self.OUTPUT_NAMES, y_pred[0]))