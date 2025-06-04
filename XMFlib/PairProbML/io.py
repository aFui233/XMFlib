import os

def load_model(model_path, model_class=None, map_location=None):
    """
    Load a PyTorch model (.pt, .pth), supports both entire model objects and state_dict.
    
    Args:
        model_path (str): Path to the model file.
        model_class (type): Optional. If your file saves only state_dict, provide the model class here.
        map_location (str or torch.device): Optional. Specify CPU/GPU device for loading the model.
    
    Returns:
        The loaded PyTorch model (or state_dict).
    
    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    ext = os.path.splitext(model_path)[-1].lower()
    if ext in ('.pt', '.pth'):
        import torch
        model_obj = torch.load(model_path, map_location=map_location, weights_only=True)
        if isinstance(model_obj, dict) and model_class is not None:
            # If the file contains only state_dict, instantiate the model structure and load parameters
            model = model_class()
            model.load_state_dict(model_obj)
            model.eval()
            return model
        else:
            # The file contains the entire model object
            model_obj.eval()
            return model_obj
    else:
        raise ValueError(f"Unsupported model file extension: {ext}")