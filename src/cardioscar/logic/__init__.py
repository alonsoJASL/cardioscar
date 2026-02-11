# src/cardioscar/logic/__init__.py
"""
Logic Public API

Exposes reconstruction logic and contracts.
"""

from cardioscar.logic.contracts import (
    PreprocessingRequest,
    PreprocessingResult,
    InferenceRequest,
    InferenceResult
)

from cardioscar.logic.reconstruction import ReconstructionLogic

from cardioscar.logic.orchestrators import (
    prepare_training_data, 
    save_preprocessing_result, 
    train_scar_model, 
    save_trained_model, 
    apply_scar_model, 
    save_inference_result,
)

__all__ = [
    # Contracts
    "PreprocessingRequest",
    "PreprocessingResult",
    "InferenceRequest",
    "InferenceResult",
    # Logic
    "ReconstructionLogic",
    # Orchestrators
    "prepare_training_data", 
    "save_preprocessing_result", 
    "train_scar_model", 
    "save_trained_model", 
    "apply_scar_model", 
    "save_inference_result",
]