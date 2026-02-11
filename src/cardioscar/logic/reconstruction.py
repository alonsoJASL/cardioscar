# src/cardioscar/logic/reconstruction.py

"""
Scar Reconstruction Logic

Stateless logic for applying trained models to meshes.
Handles coordinate normalization, MC Dropout inference, and uncertainty estimation.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from cardioscar.engines import BayesianNN
from cardioscar.logic.contracts import InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)


class ReconstructionLogic:
    """
    Stateless logic for scar reconstruction inference.
    
    This class handles:
    - Loading trained model checkpoints
    - Coordinate normalization using saved scaler parameters
    - MC Dropout inference with uncertainty quantification
    - Batch processing for memory efficiency
    
    Example:
        >>> logic = ReconstructionLogic()
        >>> request = InferenceRequest(
        ...     model_checkpoint_path=Path("model.pth"),
        ...     mesh_path=Path("mesh.vtk"),
        ...     mc_samples=10
        ... )
        >>> result = logic.run_inference(request, mesh_coords, device)
    """
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        device: torch.device
    ) -> Tuple[BayesianNN, np.ndarray, np.ndarray, dict]:
        """
        Load trained model and metadata from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: torch.device to load model onto
        
        Returns:
            Tuple of (model, scaler_min, scaler_max, metadata)
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            KeyError: If checkpoint missing required keys
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'hyperparameters', 'dataset_info']
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")
        
        # Initialize model
        dropout_rate = checkpoint['hyperparameters']['dropout_rate']
        model = BayesianNN(dropout_rate=dropout_rate).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get normalization parameters
        scaler_min = checkpoint['dataset_info']['scaler_min']
        scaler_max = checkpoint['dataset_info']['scaler_max']
        
        metadata = {
            'training_nodes': checkpoint['dataset_info']['n_nodes'],
            'training_groups': checkpoint['dataset_info']['n_groups'],
            'dropout_rate': dropout_rate
        }
        
        logger.info("Model loaded successfully")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Training nodes: {metadata['training_nodes']}")
        logger.info(f"  Training groups: {metadata['training_groups']}")
        
        return model, scaler_min, scaler_max, metadata
    
    def normalize_coordinates(
        self,
        coords: np.ndarray,
        scaler_min: np.ndarray,
        scaler_max: np.ndarray
    ) -> np.ndarray:
        """
        Normalize coordinates using saved scaler parameters.
        
        Args:
            coords: (N, 3) raw coordinates
            scaler_min: (3,) minimum values from training
            scaler_max: (3,) maximum values from training
        
        Returns:
            (N, 3) normalized coordinates in [0, 1]
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Reconstruct MinMaxScaler
        scaler = MinMaxScaler()
        scaler.data_min_ = scaler_min
        scaler.data_max_ = scaler_max
        scaler.data_range_ = scaler_max - scaler_min
        scaler.scale_ = 1.0 / scaler.data_range_
        scaler.min_ = -scaler_min * scaler.scale_
        scaler.n_samples_seen_ = 1
        scaler.n_features_in_ = 3
        
        normalized = scaler.transform(coords)
        return normalized
    
    def predict_with_uncertainty(
        self,
        model: BayesianNN,
        coords: torch.Tensor,
        mc_samples: int = 10,
        batch_size: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scar probabilities with uncertainty estimation.
        
        Args:
            model: Trained BayesianNN
            coords: (N, 3) normalized coordinates (torch tensor)
            mc_samples: Number of MC Dropout samples
            batch_size: Process in batches to avoid memory issues
        
        Returns:
            Tuple of (mean_predictions, std_predictions)
            - mean_predictions: (N,) mean scar probability
            - std_predictions: (N,) uncertainty (standard deviation)
        """
        logger.info(f"Running inference with {mc_samples} MC samples...")
        
        model.enable_dropout()  # Enable dropout for uncertainty
        
        n_nodes = len(coords)
        all_predictions = []
        
        with torch.no_grad():
            # MC Dropout sampling
            for mc_iter in range(mc_samples):
                batch_predictions = []
                
                # Process in batches
                for batch_start in range(0, n_nodes, batch_size):
                    batch_end = min(batch_start + batch_size, n_nodes)
                    batch_coords = coords[batch_start:batch_end]
                    
                    batch_pred = model(batch_coords).cpu().numpy().squeeze()
                    batch_predictions.append(batch_pred)
                
                # Concatenate batch results
                mc_pred = np.concatenate(batch_predictions)
                all_predictions.append(mc_pred)
                
                if (mc_iter + 1) % 5 == 0 or (mc_iter + 1) == mc_samples:
                    logger.info(f"  Completed {mc_iter + 1}/{mc_samples} samples")
        
        # Stack predictions: (mc_samples, n_nodes)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute statistics
        mean_pred = all_predictions.mean(axis=0)
        std_pred = all_predictions.std(axis=0)
        
        logger.info(f"Inference complete:")
        logger.info(f"  Mean scar probability: {mean_pred.mean():.3f} ± {mean_pred.std():.3f}")
        logger.info(f"  Mean uncertainty (std): {std_pred.mean():.3f}")
        logger.info(f"  Max uncertainty: {std_pred.max():.3f}")
        
        return mean_pred, std_pred
    
    def run_inference(
        self,
        request: InferenceRequest,
        mesh_coords: np.ndarray,
        device: torch.device
    ) -> InferenceResult:
        """
        Complete inference workflow.
        
        Args:
            request: InferenceRequest with paths and parameters
            mesh_coords: (N, 3) mesh node coordinates
            device: torch.device
        
        Returns:
            InferenceResult with predictions and metadata
        """
        # 1. Load model
        model, scaler_min, scaler_max, metadata = self.load_checkpoint(
            request.model_checkpoint_path,
            device
        )
        
        # 2. Normalize coordinates
        logger.info("Normalizing coordinates...")
        normalized_coords = self.normalize_coordinates(
            mesh_coords,
            scaler_min,
            scaler_max
        )
        coords_tensor = torch.from_numpy(normalized_coords).float().to(device)
        
        # 3. Predict
        mean_pred, std_pred = self.predict_with_uncertainty(
            model=model,
            coords=coords_tensor,
            mc_samples=request.mc_samples,
            batch_size=request.batch_size
        )
        
        # 4. Optional thresholding
        binary_pred = None
        if request.threshold is not None:
            binary_pred = (mean_pred >= request.threshold).astype(np.float32)
            n_scar = binary_pred.sum()
            pct_scar = 100 * n_scar / len(mesh_coords)
            logger.info(
                f"Threshold {request.threshold}: "
                f"{n_scar:.0f} nodes ({pct_scar:.1f}%) marked as scar"
            )
        
        return InferenceResult(
            mean_predictions=mean_pred,
            std_predictions=std_pred,
            binary_predictions=binary_pred,
            n_nodes=len(mesh_coords),
            mean_scar_probability=float(mean_pred.mean()),
            mean_uncertainty=float(std_pred.mean())
        )