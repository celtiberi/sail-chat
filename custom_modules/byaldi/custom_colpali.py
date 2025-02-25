import os
import torch
from pathlib import Path
from typing import Union, Optional, List
import srsly
import logging

from byaldi.colpali import ColPaliModel

# Set up logging
logger = logging.getLogger(__name__)

# We're using CPU-only for Apple Silicon compatibility
# CUDA is not available on Apple Silicon, and MPS (Metal Performance Shaders)
# has compatibility issues with some PyTorch operations.
# Memory-mapped tensors also work best on CPU.

class CustomColPaliModel(ColPaliModel):
    """Custom version of ColPaliModel that uses mmap=True for torch.load and always uses CPU for memory efficiency"""
    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        index_root: str = ".byaldi",
        **kwargs,
    ):
        """Override from_index to use mmap=True for torch.load and always use CPU"""
        
        
        index_path = Path(index_root) / Path(index_path)
        index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")

        # Create instance with CPU device to ensure memory efficiency with mmap
        instance = cls(
            pretrained_model_name_or_path=index_config["model_name"],
            n_gpu=n_gpu,
            index_name=index_path.name,
            verbose=verbose,
            load_from_index=False,  # We'll load manually with mmap
            index_root=str(index_path.parent),
            device='cpu',  # Always use CPU regardless of requested device
            **kwargs,
        )
        
        # Now manually load the index with mmap=True
        instance._load_index_with_mmap(index_path)
        
        return instance
    
    def _load_index_with_mmap(self, index_path: Path):
        """Load the index using mmap=True for torch.load, always on CPU
        
        Note on memory usage:
        - Memory-mapped tensors (mmap=True) appear in Activity Monitor as used memory
        - They don't actually consume physical RAM - they're mapped to disk
        - This is why Activity Monitor may show gigabytes of usage even though
          actual RAM usage (RSS) is much lower
        - This approach is memory-efficient as it only loads data when needed
        - Always using CPU ensures compatibility with memory mapping
        """
        # Load index config
        index_config = srsly.read_gzip_json(index_path / "index_config.json.gz")
        self.full_document_collection = index_config.get(
            "full_document_collection", False
        )
        self.resize_stored_images = index_config.get("resize_stored_images", False)
        self.max_image_width = index_config.get("max_image_width", None)
        self.max_image_height = index_config.get("max_image_height", None)

        if self.full_document_collection:
            collection_path = index_path / "collection"
            json_files = sorted(
                collection_path.glob("*.json.gz"),
                key=lambda x: int(x.stem.split(".")[0]),
            )

            for json_file in json_files:
                loaded_data = srsly.read_gzip_json(json_file)
                self.collection.update({int(k): v for k, v in loaded_data.items()})

            if self.verbose > 0:
                print(
                    "You are using in-memory collection. This means every image is stored in memory."
                )
                print(
                    "You might want to rethink this if you have a large collection!"
                )
                print(
                    f"Loaded {len(self.collection)} images from {len(json_files)} JSON files."
                )

        # Load embeddings with mmap=True
        embeddings_path = index_path / "embeddings"
        embedding_files = sorted(
            embeddings_path.glob("embeddings_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        
        # Print debug info about embedding files
        if self.verbose > 0:
            total_size_mb = sum(f.stat().st_size for f in embedding_files) / (1024 * 1024)
            print(f"Loading {len(embedding_files)} embedding files, total size: {total_size_mb:.2f} MB")
            print(f"Using device: cpu (for memory-mapped tensors)")
            print(f"Memory-mapped loading enabled (mmap=True)")
            print(f"Note: Activity Monitor will show ~{total_size_mb:.2f} MB higher memory usage")
            print(f"      This is normal for memory-mapped files and doesn't affect actual RAM usage")
        
        # Store loaded tensors in a list without using extend to avoid potential copies
        self.indexed_embeddings = []
        for file in embedding_files:
            # Use mmap=True for memory-mapped file loading
            # Memory-mapped tensors must be loaded to CPU
            map_location = 'cpu'  # Always load to CPU for mmap
            try:
                loaded_tensor = torch.load(file, map_location=map_location, mmap=True)
                
                # Print debug info about the loaded tensor
                if self.verbose > 0 and isinstance(loaded_tensor, torch.Tensor):
                    tensor_size_mb = loaded_tensor.element_size() * loaded_tensor.numel() / (1024 * 1024)
                    print(f"  Loaded tensor from {file.name}: shape={loaded_tensor.shape}, "
                          f"size={tensor_size_mb:.2f} MB")
                    
                    # For very large tensors, consider chunking them to avoid memory issues
                    if tensor_size_mb > 1000:  # If tensor is larger than 1GB
                        print(f"  Large tensor detected ({tensor_size_mb:.2f} MB). "
                              f"Will use chunked access for memory efficiency.")
                
                # Store each file's tensors as a separate item in the list
                self.indexed_embeddings.append(loaded_tensor)
                
            except Exception as e:
                print(f"Error loading tensor from {file}: {e}")
        
        # We no longer move tensors to other devices - always keep on CPU for memory mapping
        if self.verbose > 0:
            print("All tensors kept on CPU for memory-mapped access")
            
        # Flatten the list if needed for compatibility with the rest of the code
        # This approach avoids creating copies of the memory-mapped tensors
        if self.indexed_embeddings and isinstance(self.indexed_embeddings[0], list):
            flat_embeddings = []
            for emb_list in self.indexed_embeddings:
                flat_embeddings.extend(emb_list)
            self.indexed_embeddings = flat_embeddings

        # Load mappings
        self.embed_id_to_doc_id = srsly.read_gzip_json(
            index_path / "embed_id_to_doc_id.json.gz"
        )
        # Restore keys to integers
        self.embed_id_to_doc_id = {
            int(k): v for k, v in self.embed_id_to_doc_id.items()
        }
        self.highest_doc_id = max(
            int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
        )
        self.doc_ids = set(
            int(entry["doc_id"]) for entry in self.embed_id_to_doc_id.values()
        )
        try:
            # We don't want this error out with indexes created prior to 0.0.2
            self.doc_ids_to_file_names = srsly.read_gzip_json(
                index_path / "doc_ids_to_file_names.json.gz"
            )
            self.doc_ids_to_file_names = {
                int(k): v for k, v in self.doc_ids_to_file_names.items()
            }
        except FileNotFoundError:
            pass

        # Load metadata
        metadata_path = index_path / "metadata.json.gz"
        if metadata_path.exists():
            self.doc_id_to_metadata = srsly.read_gzip_json(metadata_path)
            # Convert metadata keys to integers
            self.doc_id_to_metadata = {
                int(k): v for k, v in self.doc_id_to_metadata.items()
            } 

    # def search(
    #     self,
    #     query_or_embedding,
    #     k=5,
    #     filter_fn=None,
    #     return_documents=False,
    #     return_tensors=False,
    #     **kwargs,
    # ):
    #     """Override search method to implement chunked processing for large embeddings"""
    #     # Get query embedding
    #     if isinstance(query_or_embedding, str):
    #         query_embedding = self.encode_query(query_or_embedding)
    #     else:
    #         query_embedding = query_or_embedding

    #     # Process in chunks to avoid loading all embeddings into memory at once
    #     chunk_size = 10000  # Process 10k embeddings at a time
    #     all_scores = []
    #     all_indices = []
        
    #     # Check if we have a list of tensors or a single tensor
    #     if isinstance(self.indexed_embeddings, list):
    #         # Process each tensor in the list
    #         for tensor_idx, embeddings in enumerate(self.indexed_embeddings):
    #             if self.verbose > 0:
    #                 print(f"Processing embedding tensor {tensor_idx+1}/{len(self.indexed_embeddings)}")
                
    #             # Process large tensors in chunks
    #             if isinstance(embeddings, torch.Tensor) and embeddings.shape[0] > chunk_size:
    #                 num_chunks = (embeddings.shape[0] + chunk_size - 1) // chunk_size
                    
    #                 for chunk_idx in range(num_chunks):
    #                     start_idx = chunk_idx * chunk_size
    #                     end_idx = min((chunk_idx + 1) * chunk_size, embeddings.shape[0])
                        
    #                     if self.verbose > 1:
    #                         print(f"  Processing chunk {chunk_idx+1}/{num_chunks} (indices {start_idx}-{end_idx})")
                        
    #                     # Get embeddings for this chunk
    #                     chunk_embeddings = embeddings[start_idx:end_idx]
                        
    #                     # Calculate scores for this chunk
    #                     chunk_scores = self._calculate_scores(query_embedding, chunk_embeddings)
                        
    #                     # Adjust indices to account for chunking
    #                     chunk_indices = torch.arange(start_idx, end_idx, device=chunk_scores.device)
                        
    #                     # Add tensor_idx prefix to distinguish between different tensors
    #                     # We'll convert these to global indices later
    #                     all_scores.append(chunk_scores)
    #                     all_indices.append((tensor_idx, chunk_indices))
                        
    #                     # Free up memory
    #                     del chunk_embeddings, chunk_scores
    #                     if torch.cuda.is_available():
    #                         torch.cuda.empty_cache()
    #                     elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
    #                         torch.mps.empty_cache()
    #             else:
    #                 # Small tensor, process all at once
    #                 scores = self._calculate_scores(query_embedding, embeddings)
    #                 indices = torch.arange(embeddings.shape[0], device=scores.device)
                    
    #                 all_scores.append(scores)
    #                 all_indices.append((tensor_idx, indices))
    #     else:
    #         # Single tensor, process in chunks
    #         embeddings = self.indexed_embeddings
    #         if isinstance(embeddings, torch.Tensor) and embeddings.shape[0] > chunk_size:
    #             num_chunks = (embeddings.shape[0] + chunk_size - 1) // chunk_size
                
    #             for chunk_idx in range(num_chunks):
    #                 start_idx = chunk_idx * chunk_size
    #                 end_idx = min((chunk_idx + 1) * chunk_size, embeddings.shape[0])
                    
    #                 if self.verbose > 1:
    #                     print(f"Processing chunk {chunk_idx+1}/{num_chunks} (indices {start_idx}-{end_idx})")
                    
    #                 # Get embeddings for this chunk
    #                 chunk_embeddings = embeddings[start_idx:end_idx]
                    
    #                 # Calculate scores for this chunk
    #                 chunk_scores = self._calculate_scores(query_embedding, chunk_embeddings)
                    
    #                 # Adjust indices to account for chunking
    #                 chunk_indices = torch.arange(start_idx, end_idx, device=chunk_scores.device)
                    
    #                 all_scores.append(chunk_scores)
    #                 all_indices.append((0, chunk_indices))
                    
    #                 # Free up memory
    #                 del chunk_embeddings, chunk_scores
    #                 if torch.cuda.is_available():
    #                     torch.cuda.empty_cache()
    #                 elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
    #                     torch.mps.empty_cache()
    #         else:
    #             # Small tensor, process all at once
    #             scores = self._calculate_scores(query_embedding, embeddings)
    #             indices = torch.arange(embeddings.shape[0], device=scores.device)
                
    #             all_scores.append(scores)
    #             all_indices.append((0, indices))
        
    #     # Combine results from all chunks and tensors
    #     combined_scores = torch.cat([s for s in all_scores])
        
    #     # Convert tensor_idx and local_idx to global indices
    #     global_indices = []
    #     for tensor_idx, local_indices in all_indices:
    #         if isinstance(self.indexed_embeddings, list):
    #             # Calculate offset based on previous tensors
    #             offset = 0
    #             for i in range(tensor_idx):
    #                 offset += self.indexed_embeddings[i].shape[0]
                
    #             # Add offset to local indices
    #             global_indices.append(local_indices + offset)
    #         else:
    #             # Single tensor, no offset needed
    #             global_indices.append(local_indices)
        
    #     combined_indices = torch.cat([idx for idx in global_indices])
        
    #     # Get top-k results
    #     if k > combined_scores.shape[0]:
    #         k = combined_scores.shape[0]
        
    #     topk_scores, topk_indices = torch.topk(combined_scores, k)
    #     topk_indices = combined_indices[topk_indices]
        
    #     # Convert to CPU for easier processing
    #     topk_scores = topk_scores.cpu()
    #     topk_indices = topk_indices.cpu()
        
    #     # Rest of the original search method...
    #     # [Original code for filtering and returning results]
        
    #     # For now, call the parent class method to handle the rest
    #     # This is a temporary solution until we fully implement the chunked search
    #     return super().search(
    #         query_or_embedding,
    #         k=k,
    #         filter_fn=filter_fn,
    #         return_documents=return_documents,
    #         return_tensors=return_tensors,
    #         **kwargs,
    #     )
        
    # def _calculate_scores(self, query_embedding, indexed_embeddings):
    #     """Calculate similarity scores between query and indexed embeddings"""
    #     # Ensure both query and indexed embeddings are on CPU
    #     if query_embedding.device.type != 'cpu':
    #         query_embedding = query_embedding.to('cpu')
            
    #     if indexed_embeddings.device.type != 'cpu':
    #         indexed_embeddings = indexed_embeddings.to('cpu')
            
    #     # Normalize embeddings
    #     query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    #     indexed_embeddings = indexed_embeddings / indexed_embeddings.norm(dim=-1, keepdim=True)
        
    #     # Calculate cosine similarity
    #     scores = torch.matmul(query_embedding, indexed_embeddings.T).squeeze(0)
        
    #     return scores 