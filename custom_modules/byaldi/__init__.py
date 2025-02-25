from byaldi.RAGModel import RAGMultiModalModel as BaseRAGMultiModalModel
from .custom_colpali import CustomColPaliModel
import torch
import os
import platform
import sys


class RAGMultiModalModel(BaseRAGMultiModalModel):
    """Override RAGMultiModalModel to use our custom ColPaliModel implementation"""
    
    @classmethod
    def from_index(cls, *args, **kwargs):
        """Override from_index to handle device selection appropriately"""
        # Check if we're on Apple Silicon
        is_apple_silicon = (
            platform.system() == "Darwin" and 
            platform.machine() == "arm64"
        )
        
        # Extract device from kwargs
        original_device = kwargs.get('device', None)
        
        # Force CPU on Apple Silicon
        if is_apple_silicon:
            if original_device not in [None, 'cpu']:
                print(f"Apple Silicon detected. Forcing CPU usage regardless of requested device '{original_device}'.")
            kwargs['device'] = 'cpu'
            return super().from_index(*args, **kwargs)
        
        # For non-Apple Silicon, use the original device or CPU
        return super().from_index(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override from_pretrained to use our custom ColPaliModel"""
        # Replace the model_cls with our custom implementation
        kwargs['model_cls'] = CustomColPaliModel
        return super().from_pretrained(*args, **kwargs) 