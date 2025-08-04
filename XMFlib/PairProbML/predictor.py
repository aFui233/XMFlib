import os
import torch
import glob
import numpy as np
from .io import load_model
from .utils import eps_kbt, parse_hidden_layers, data_norm, data_denorm
from .models import MLP, MLP_2nn

class PairProbPredictor:
    SUPPORTED_FACETS = ['100', '111']
    OUTPUT_NAMES = ['vacancy_pair', 'species_pair', 'species_vacancy_pair']

    def __init__(self, model_dir=None, data_dir=None):
        # Set model directory; use default if not provided
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        self.model_dir = model_dir
        self.data_dir = data_dir

    def predict(self, facet, interaction_energy, temperature, main_coverage,
                model_type='mlp', task='a2p'):
        # Ensure facet is string
        facet = str(facet)
        if facet not in self.SUPPORTED_FACETS:
            raise ValueError(f"Facet '{facet}' not supported. Supported: {self.SUPPORTED_FACETS}")

        target_list = ['pee', 'paa']
        y_pred_list = []

        for target in target_list:
            dataset_file = f'{facet}_{target}_cleaned.npy'
            dataset_path = os.path.join(self.data_dir, dataset_file)
            dataset = np.load(dataset_path)

            model_file_pattern = f'{model_type}_{task}_{facet}_{target}_*.pt'
            model_files = glob.glob(os.path.join(self.model_dir, model_file_pattern))
            if not model_files:
                raise FileNotFoundError(f"Model file matching '{model_file_pattern}' not found in directory '{self.model_dir}'.")
            model_path = model_files[0]  # Get the first matching file
            hidden_layers = parse_hidden_layers(model_path)

            mlp_model = MLP(hidden_layers, num_inputs=2, num_outputs=1)
            mlp_model.load_state_dict(torch.load(model_path))
            mlp_model.eval()
            # model = load_model(model_path, model_class=mlp_model)

            dimless_eps = eps_kbt(interaction_energy, temperature)

            X = [[dimless_eps, main_coverage]]
            X_norm = data_norm(X, dataset)
            X_tensor = torch.tensor(X_norm, dtype=torch.float32)

            with torch.no_grad():
                log_y_pred_norm = mlp_model(X_tensor).squeeze().numpy()

            log_y_pred = data_denorm(log_y_pred_norm, dataset)
            y_pred = np.exp(log_y_pred)
            y_pred_list.append(y_pred)

        y_pee_pred = y_pred_list[0]
        y_paa_pred = y_pred_list[1]
        y_pae_pred = 1 - y_pee_pred - y_paa_pred

        y_pred_list.append(y_pae_pred)
        y_pred_list = [float(x) for x in y_pred_list]

        return y_pred_list

    def predict_2nn(self, facet, interaction_energy_1nn, interaction_energy_2nn,
                    temperature, main_coverage, model_type='mlp', task='a2p'):
        # Ensure facet is string
        facet = str(facet)
        if facet not in self.SUPPORTED_FACETS:
            raise ValueError(f"Facet '{facet}' not supported. Supported: {self.SUPPORTED_FACETS}")

        # Build the model file name and path
        model_file = f'{model_type}_{task}_{facet}_2nn.pth'
        model_path = os.path.join(self.model_dir, model_file)

        # Load the trained model
        model = load_model(model_path, model_class=MLP_2nn)

        # Convert interaction energies to dimensionless form
        dimless_eps_1nn = eps_kbt(interaction_energy_1nn, temperature)
        dimless_eps_2nn = eps_kbt(interaction_energy_2nn, temperature)

        # Prepare input features (feature order must match training)
        X = [[dimless_eps_1nn, dimless_eps_2nn, main_coverage]]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            y_pred = model(X_tensor).numpy()

        # Return predictions as a dictionary
        return y_pred[0].tolist()