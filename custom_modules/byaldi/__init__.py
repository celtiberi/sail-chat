import platform
import logging
import os
import sys

# Set up logging
logger = logging.getLogger(__name__)

# Detect system type
is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith("arm")

# Import appropriate implementation based on platform
if is_apple_silicon:
    logger.info("Running on Apple Silicon, using CPU-optimized implementation")
    # Import our custom implementation for Apple Silicon
    from .custom_colpali import CustomColPaliModel
    from byaldi.objects import Result
    import torch
    
    # Monkey patch torch.cuda to prevent CUDA initialization errors on Apple Silicon
    original_lazy_init = torch.cuda._lazy_init

    def safe_lazy_init():
        """Replacement for torch.cuda._lazy_init that doesn't raise an error"""
        logger.warning("Prevented CUDA initialization error - using CPU instead")
        return False

    # Apply the monkey patch
    torch.cuda._lazy_init = safe_lazy_init

    # Also patch torch.cuda.is_available to always return False on Apple Silicon
    original_is_available = torch.cuda.is_available

    def safe_is_available():
        """Always return False for CUDA availability on Apple Silicon"""
        return False

    # Apply the patch
    torch.cuda.is_available = safe_is_available
    
    # Set environment variables to ensure CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Create our custom RAGMultiModalModel for Apple Silicon
    from byaldi.RAGModel import RAGMultiModalModel as BaseRAGMultiModalModel
    
    class RAGMultiModalModel(BaseRAGMultiModalModel):
        """Apple Silicon optimized version of RAGMultiModalModel that forces CPU usage with memory-mapped tensors"""
        
        @classmethod
        def from_index(cls, index_path, **kwargs):
            """Override from_index to force CPU usage with our custom model"""
            logger.info("Using CPU-optimized implementation with memory-mapped tensors for Apple Silicon")
            
            # Force CPU usage
            kwargs['device'] = 'cpu'
            
            # Remove model_cls if it exists in kwargs
            if 'model_cls' in kwargs:
                del kwargs['model_cls']
            
            # Call the parent class's from_index method
            instance = super().from_index(index_path, **kwargs)
            
            # Replace the model with our custom model
            # This is a workaround since we can't pass model_cls directly
            logger.info("Initializing CustomColPaliModel for Apple Silicon")
            
            # Create a new kwargs dict for CustomColPaliModel to avoid parameter conflicts
            custom_kwargs = {
                'verbose': kwargs.get('verbose', 1)
            }
            
            instance.model = CustomColPaliModel.from_index(
                index_path=index_path,
                **custom_kwargs
            )
            
            return instance
        
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            """Override from_pretrained to use our custom ColPaliModel"""
            logger.info("Using CPU-optimized implementation with memory-mapped tensors for Apple Silicon")
            
            # Force CPU usage
            kwargs['device'] = 'cpu'
            
            # Remove model_cls if it exists in kwargs
            if 'model_cls' in kwargs:
                del kwargs['model_cls']
            
            # Call the parent class's from_pretrained method
            instance = super().from_pretrained(model_name, **kwargs)
            
            # Replace the model with our custom model
            # This is a workaround since we can't pass model_cls directly
            logger.info("Initializing CustomColPaliModel for Apple Silicon")
            
            # Create a new kwargs dict for CustomColPaliModel to avoid parameter conflicts
            custom_kwargs = {
                'verbose': kwargs.get('verbose', 1)
            }
            
            instance.model = CustomColPaliModel.from_pretrained(
                model_name,
                **custom_kwargs
            )
            
            return instance
else:
    logger.info("Running on standard hardware, using default implementation")
    # Import standard implementation for non-Apple Silicon
    from byaldi.RAGModel import RAGMultiModalModel
    from byaldi.objects import Result 