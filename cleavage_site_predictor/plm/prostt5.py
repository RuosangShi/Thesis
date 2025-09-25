'''Embeddings from ProstT5'''

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
import numpy as np
import os
import ast
from typing import Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structural_analyzer.cache_manager import CacheManager

class ProstT5(CacheManager):
    def __init__(self, max_cache_size=100, cache_dir="structure_temp/prostt5_cache", 
                 use_disk_cache=True, use_ram_cache=True, uniprot_timeout=30):
        super().__init__(cache_dir, use_disk_cache, use_ram_cache)
        self.uniprot_timeout = uniprot_timeout
        # Optimize for Mac M4 chip - prioritize MPS over CUDA and CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using Mac M4 GPU (MPS) for ProstT5")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f"Using CUDA GPU for ProstT5")
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU for ProstT5")

        # Load the tokenizer
        # tokenizer object with special tokens like <AA2fold> and <fold2AA>
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

        # Initialize model as None for lazy loading  
        self._encoder_model_instance = None
        
        # Enhanced cache configuration
        self.max_cache_size = max_cache_size
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache
        self.use_ram_cache = use_ram_cache
        
        # Create disk cache directory
        if self.use_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(os.path.join(cache_dir, 'embeddings'), exist_ok=True)
        
        # Initialize caches
        if self.use_ram_cache:
            self.embedding_cache = {}  # cache for sequence embeddings
        else:
            self.embedding_cache = {}

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key using MD5 hash"""
        import hashlib
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str, cache_type: str):
        """Get data from two-level cache (RAM -> disk)"""
        # 1. Check RAM cache (fastest)
        if self.use_ram_cache:
            if cache_type == 'embeddings' and cache_key in self.embedding_cache:
                print(f"RAM cache hit: embeddings_{cache_key[:8]}...")
                return self.embedding_cache[cache_key]
        
        # 2. Check disk cache (medium speed)
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
        """Control the cache size, when it exceeds the limit, remove the earliest added item"""
        if len(cache_dict) > self.max_cache_size:
            # remove 10% of the cache items
            items_to_remove = int(self.max_cache_size * 0.1)
            for _ in range(items_to_remove):
                if cache_dict:
                    cache_dict.pop(next(iter(cache_dict)))

    @property
    # Use lazy loading to load the model
    def _encoder_model(self):
        if self._encoder_model_instance is None:
            self._encoder_model_instance = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)
            # MPS and CPU require float32, only CUDA supports half precision
            if self.device.type in ['cpu', 'mps']:
                self._encoder_model_instance.float()
                if self.device.type == 'mps':
                    print("Using float32 precision for MPS compatibility")
            else:
                self._encoder_model_instance.half()
                print("Using half precision on CUDA")
        return self._encoder_model_instance


    def format_sequences(self, sequences=["PRTEINO", "strct"]):
        '''
        Prepare the protein sequences/structures as a list.
        Amino acid sequences are expected to be upper-case ("PRTEINO")
        while 3Di-sequences need to be lower-case ("strctr").

        example:
        sequence = ["PRTEINO", "strct"]
        sequence_formatted = ['<AA2fold> P R T E I N O', '<fold2AA> s t r c t']
        '''
        # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) 
        # and introduce white-space between all sequences (AAs and 3Di)
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

        # The direction of the translation is indicated by two special tokens:
        # "<AA2fold>": from AAs to 3Di (or if embed AAs)
        # "<fold2AA>": from 3Di to AAs (or if embed 3Di)
        
        sequence_formatted = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                      for s in sequences
                    ]
        return sequence_formatted

    def tokenize_sequences(self, sequences_formatted):
        '''
        Tokenize the sequences.
        e.g. if sequence_formatted = ['<AA2fold> P R T E I N O', '<fold2AA> s t r c t']
        Max sequence length = 8, because: <AA2fold> counts as 1 token
        "P R T E I N O" has 7 tokens → total = 8
        '''
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(sequences_formatted,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(self.device)
        return ids

    def _get_embeddings(self, sequences, remove_padding=True):
        '''
        Get the embeddings. remove_padding: if True, remove the padding tokens

        embedding_repr.last_hidden_state
        # shape: (batch_size, sequence_length (include padding), embedding_dim)

        return: protein_emd, residue_emb
        protein_emd: tensor, shape (1024)
        if remove_padding is False:
            residue_emb: tensor, shape (batch_size, sequence_length (include padding), embedding_dim)
        if remove_padding is True:
            residue_emb: List[tensor], each tensor shape (sequence_length (exclude padding), embedding_dim)

        if input is 2 sequences, length is 7 and 8 (after removing padding, the output could be:
        shape: (2, max_len, 1024)  # max_len is at least 8(max length)

        So if the input sequence_examples is [AA_sequence, 3Di_sequence]
        => embedding_repr.last_hidden_state[0, 1:sequence_length] is AA_sequence embedding
        => embedding_repr.last_hidden_state[1, 1:sequence_length] is 3Di_sequence embedding
        '''
        # Generate cache key
        cache_key = self._generate_cache_key(sequences=tuple(sequences), remove_padding=remove_padding)
        
        # Check two-level cache
        cached_data = self._get_cached_data(cache_key, 'embeddings')
        if cached_data is not None:
            return cached_data
        
        sequence_formatted = self.format_sequences(sequences)
        ids = self.tokenize_sequences(sequence_formatted)
        # generate embeddings
        with torch.no_grad():
            embedding_repr = self._encoder_model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )
        # Get the lengths from attention mask (excluding prefix and </s>)
        lengths = ids['attention_mask'].sum(dim=1) - 2  # subtract prefix + </s>
        if remove_padding: # remove padding tokens
            residue_embeddings = []
            for i in range(len(sequences)): # how many sequences
                # Each emb's shape is (seq_len, 1024), where seq_len is the actual sequence length
                emb = embedding_repr.last_hidden_state[i, 1:1+lengths[i]] # shape:  (sequence_length (exclude padding), embedding_dim)
                residue_embeddings.append(emb) 
            # calculate the average embedding for each sequence
            # shape: (n_sequences, 1024) where n_sequences is the number of input sequences
            protein_embeddings = torch.stack([emb.mean(dim=0) for emb in residue_embeddings]) # average on the sequence length dimension
        else: # do not remove padding tokens, residue_embeddings is a tensor, use mean()
            print("Not removing padding tokens...")
            # remove the prefix and </s> tokens
            residue_embeddings = embedding_repr.last_hidden_state[:, 1:-1] # shape: (batch_size, sequence_length (include padding), embedding_dim)
            protein_embeddings = residue_embeddings.mean(dim=1) # avergae on the sequence length dimension => shape: (batch_size, embedding_dim)
        
        # Cache the embeddings with two-level cache
        result = (protein_embeddings, residue_embeddings)
        self._cache_data(cache_key, result, 'embeddings')
        return result

    def get_full_sequence_embeddings(self, sequence):
        """
        Extract embeddings for the entire protein sequence and cache them.
        This method extracts all residue embeddings at once and caches them
        for efficient window-based access later.
        
        Args:
            sequence (str): Full protein sequence
            
        Returns:
            torch.Tensor: Residue embeddings, shape (seq_len, 1024)
        """
        # Generate cache key for full sequence
        cache_key = self._generate_cache_key(sequence=sequence, full_sequence=True)
        
        # Check cache first
        cached_data = self._get_cached_data(cache_key, 'embeddings')
        if cached_data is not None:
            return cached_data
        
        # Extract embeddings for full sequence
        print(f"Computing full sequence embeddings for sequence of length {len(sequence)}...")
        protein_emb, residue_emb = self._get_embeddings([sequence], remove_padding=True)
        
        # residue_emb is a list with one element (tensor of shape (seq_len, 1024))
        #full_residue_embeddings = residue_emb[0].cpu()  # Move to CPU to save GPU memory
        
        result = (protein_emb, residue_emb)
        # Cache the full residue embeddings
        self._cache_data(cache_key, result, 'embeddings')
        
        return result




