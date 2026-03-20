import os
import glob
import torch
import numpy as np
from .utils import eps_kbt, parse_hidden_layers, data_norm, data_denorm
from .models import MLP


class CovPredictor:
    SUPPORTED_FACETS = ["100", "111"]
    OUTPUT_NAMES = ["A_Coverage", "E_Coverage"]

    def __init__(self, model_dir=None, data_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "dataset")
        self.model_dir = model_dir
        self.data_dir = data_dir

    def predict(
        self,
        facet,
        interaction_energy,
        adsorption_energy,
        temperature,
        model_type="mlp",
        task="e2c",
    ):
        facet = str(facet)
        if facet not in self.SUPPORTED_FACETS:
            raise ValueError(
                f"Facet '{facet}' not supported. Supported: {self.SUPPORTED_FACETS}"
            )

        dataset_file = f"{facet}_e2c_cleaned.npy"
        dataset_path = os.path.join(self.data_dir, dataset_file)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        dataset = np.load(dataset_path)

        model_file_pattern = f"{model_type}_{task}_{facet}_*.pt"
        model_files = glob.glob(os.path.join(self.model_dir, model_file_pattern))
        if not model_files:
            raise FileNotFoundError(
                f"Model file matching '{model_file_pattern}' not found in '{self.model_dir}'."
            )
        model_path = model_files[0]

        hidden_layers = parse_hidden_layers(model_path)
        mlp_model = MLP(hidden_layers, num_inputs=2, num_outputs=1)
        mlp_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        mlp_model.eval()

        dimless_eps = eps_kbt(interaction_energy, temperature)
        dimless_eads = eps_kbt(adsorption_energy, temperature)

        X = [[dimless_eps, dimless_eads]]
        X_norm = data_norm(np.array(X), dataset)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        with torch.no_grad():
            y_pred_norm = mlp_model(X_tensor).squeeze().numpy()

        theta_a = float(data_denorm(y_pred_norm, dataset))
        theta_e = float(1.0 - theta_a)

        return [theta_a, theta_e]