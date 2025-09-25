'''
ESM-2 feature extraction (1280 dim):

esm.pretrained.	#layers	#params	Dataset	Embedding Dim
esm2_t33_650M_UR50D	33	650M	UR50/D 2021_04	1280

https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

https://github.com/facebookresearch/esm

pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
'''

import torch
import numpy as np
import pandas as pd
import os
import sys
from typing import Any, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structural_analyzer.cache_manager import CacheManager

import esm


class ESM2(CacheManager):
    """
    ESM-2 protein language model implementation following ProtT5 pattern
    """
    
    def __init__(self, 
                 model_name: str = "esm2_t33_650M_UR50D",
                 max_cache_size: int = 100, 
                 cache_dir: str = "structure_temp/esm2_cache", 
                 use_disk_cache: bool = True, 
                 use_ram_cache: bool = True, 
                 uniprot_timeout: int = 30):
        """
        Initialize ESM-2 model with caching capabilities
        
        Args:
            model_name: ESM-2 model variant to use
            max_cache_size: Maximum number of items in RAM cache
            cache_dir: Directory for disk cache
            use_disk_cache: Enable disk caching
            use_ram_cache: Enable RAM caching
            uniprot_timeout: Timeout for UniProt requests
        """
        super().__init__(cache_dir, use_disk_cache, use_ram_cache)
        
        self.model_name = model_name
        self.uniprot_timeout = uniprot_timeout
        
        # Device selection - prioritize MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using Apple MPS GPU for ESM-2")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA GPU for ESM-2")
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU for ESM-2")
        
        # Lazy loading for model and alphabet
        self._model_instance = None
        self._alphabet_instance = None
        self._batch_converter_instance = None
        
        # Cache configuration
        self.max_cache_size = max_cache_size
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache
        self.use_ram_cache = use_ram_cache
        
        # Create cache directories
        if self.use_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(os.path.join(cache_dir, 'embeddings'), exist_ok=True)
        
        # Initialize caches
        if self.use_ram_cache:
            self.embedding_cache = {}
        else:
            self.embedding_cache = {}
            
        print(f"ESM-2 initialized with model: {model_name}")
        print(f"Device: {self.device}")
    
    @property
    def model(self):
        """Lazy load ESM-2 model"""
        if self._model_instance is None:
            print(f"Loading ESM-2 model: {self.model_name}")
            
            if self.model_name == "esm2_t33_650M_UR50D":
                self._model_instance, self._alphabet_instance = esm.pretrained.esm2_t33_650M_UR50D()
            else:
               print(f"Model {self.model_name} not found")
            
            self._model_instance = self._model_instance.to(self.device)
            self._model_instance.eval()  # disable dropout for deterministic results
            
            if self.device.type in ['cpu', 'mps']:
                self._model_instance.float()
                if self.device.type == 'mps':
                    print("Using float32 precision for MPS compatibility")
            else:
                self._model_instance.half()
                print("Using half precision on CUDA")
        
        return self._model_instance
    
    @property
    def alphabet(self):
        """Get ESM-2 alphabet"""
        if self._alphabet_instance is None:
            # This will trigger model loading which also loads alphabet
            _ = self.model
        return self._alphabet_instance
    
    @property
    def batch_converter(self):
        """Get batch converter"""
        if self._batch_converter_instance is None:
            self._batch_converter_instance = self.alphabet.get_batch_converter()
        return self._batch_converter_instance
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key using MD5 hash"""
        import hashlib
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str, cache_type: str):
        """Get data from two-level cache (RAM -> disk)"""
        # Check RAM cache first
        if self.use_ram_cache:
            if cache_type == 'embeddings' and cache_key in self.embedding_cache:
                print(f"RAM cache hit: embeddings_{cache_key[:8]}...")
                return self.embedding_cache[cache_key]
        
        # Check disk cache
        if self.use_disk_cache:
            disk_data = self._load_from_disk(cache_key, cache_type)
            if disk_data is not None:
                print(f"Disk cache hit: {cache_type}_{cache_key[:8]}...")
                
                # Promote to RAM cache
                if self.use_ram_cache:
                    if cache_type == 'embeddings':
                        self.embedding_cache[cache_key] = disk_data
                        self._manage_cache_size(self.embedding_cache)
                
                return disk_data
        
        return None
    
    def _cache_data(self, cache_key: str, data: Any, cache_type: str) -> None:
        """Save data to two-level cache"""
        # Save to RAM cache
        if self.use_ram_cache:
            if cache_type == 'embeddings':
                self.embedding_cache[cache_key] = data
                self._manage_cache_size(self.embedding_cache)
        
        # Save to disk cache
        self._save_to_disk(cache_key, data, cache_type)
    
    def _save_to_disk(self, cache_key: str, data: Any, cache_type: str) -> bool:
        """Save data to disk cache"""
        if not self.use_disk_cache:
            return False
        
        try:
            import pickle
            cache_subdir = os.path.join(self.cache_dir, cache_type)
            cache_file = os.path.join(cache_subdir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")
            return False
    
    def _load_from_disk(self, cache_key: str, cache_type: str):
        """Load data from disk cache"""
        if not self.use_disk_cache:
            return None
        
        try:
            import pickle
            cache_subdir = os.path.join(self.cache_dir, cache_type)
            cache_file = os.path.join(cache_subdir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")
        return None
    
    def _manage_cache_size(self, cache_dict):
        """Control cache size by removing oldest items when limit exceeded"""
        if len(cache_dict) > self.max_cache_size:
            items_to_remove = int(self.max_cache_size * 0.1)
            for _ in range(items_to_remove):
                if cache_dict:
                    cache_dict.pop(next(iter(cache_dict)))
    
    def _get_embeddings(self, sequences: List[str], remove_padding: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Get ESM-2 embeddings for protein sequences
        
        Args:
            sequences: List of protein sequences
            remove_padding: Whether to remove padding tokens
            
        Returns:
            Tuple of (protein_embeddings, residue_embeddings)
            - protein_embeddings: Average embedding per sequence, shape (n_sequences, 1280)
            - residue_embeddings: List of per-residue embeddings, each shape (seq_len, 1280)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            sequences=tuple(sequences), 
            remove_padding=remove_padding,
            model_name=self.model_name
        )
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 'embeddings')
        if cached_data is not None:
            return cached_data
        
        # Prepare data for ESM-2 following official tutorial pattern
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            
        # Get representations from last layer (layer 33)
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
        token_representations = results["representations"][33]  # Shape: (batch_size, seq_len+2, 1280)
        
        if remove_padding:
            residue_embeddings = []
            protein_embeddings_list = []
            
            for i, tokens_len in enumerate(batch_lens):
                # Extract per-residue representations following official pattern
                # token 0 is beginning-of-sequence, token tokens_len-1 is end-of-sequence
                seq_repr = token_representations[i, 1:tokens_len-1]  # Remove <cls> and <eos>
                residue_embeddings.append(seq_repr)
                
                # Generate per-sequence representation via averaging
                protein_emb = seq_repr.mean(dim=0)
                protein_embeddings_list.append(protein_emb)
            
            protein_embeddings = torch.stack(protein_embeddings_list)
            
        else:
            # Keep padding, remove special tokens
            residue_embeddings = token_representations[:, 1:-1]  # Remove <cls> and <eos>
            protein_embeddings = residue_embeddings.mean(dim=1)  # Average over sequence length
        
        # Cache results
        result = (protein_embeddings, residue_embeddings)
        self._cache_data(cache_key, result, 'embeddings')
        
        return result
    
    def get_full_sequence_embeddings(self, sequence: str) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract embeddings for full protein sequence with caching
        
        Args:
            sequence: Full protein sequence
            
        Returns:
            Tuple of (protein_embedding, residue_embeddings)
            - protein_embedding: Average embedding for sequence, shape (1, 1280)
            - residue_embeddings: List with one tensor of shape (seq_len, 1280)
        """
        # Generate cache key for full sequence
        cache_key = self._generate_cache_key(
            sequence=sequence, 
            full_sequence=True,
            model_name=self.model_name
        )
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 'embeddings')
        if cached_data is not None:
            return cached_data
        
        # Extract embeddings for full sequence
        print(f"Computing ESM-2 embeddings for sequence of length {len(sequence)}...")
        protein_emb, residue_emb = self._get_embeddings([sequence], remove_padding=True)
        
        # Cache and return
        result = (protein_emb, residue_emb)
        self._cache_data(cache_key, result, 'embeddings')
        
        return result
    
    def format_sequences(self, sequences: List[str]) -> List[str]:
        """
        Format sequences for ESM-2 (no special formatting needed like ProtT5)
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            List of formatted sequences (same as input for ESM-2)
        """
        # ESM-2 doesn't need special formatting like ProtT5
        # Just clean any ambiguous amino acids
        import re
        cleaned_sequences = []
        for seq in sequences:
            # Replace ambiguous amino acids with X
            cleaned_seq = re.sub(r"[UZOB]", "X", seq.upper())
            cleaned_sequences.append(cleaned_seq)
        
        return cleaned_sequences
