"""
PLM (Protein Language Model) Profiler
Integrates protein language model embeddings with the structural analyzer cache system
Similar architecture to StructuralProfiler with base and child classes
"""

import pandas as pd
import numpy as np
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .cache_manager import CacheManager
from cleavage_site_predictor.plm.prostt5 import ProstT5


class PLMProfiler(CacheManager):
    """
    Base PLM Profiler class that extends CacheManager for consistent caching behavior
    with other structural analyzers. Similar to StructuralProfiler pattern.
    """
    
    def __init__(self, 
                 cache_dir: str = "structure_temp/plm_cache",
                 use_disk_cache: bool = True, 
                 use_ram_cache: bool = True,
                 uniprot_timeout: int = 30):
        """
        Initialize PLM Profiler with caching
        
        Args:
            cache_dir: Directory for cache storage
            use_disk_cache: Enable disk caching
            use_ram_cache: Enable RAM caching
            uniprot_timeout: Timeout for UniProt API requests (seconds)
        """
        super().__init__(cache_dir, use_disk_cache, use_ram_cache)
        self.uniprot_timeout = uniprot_timeout
        
        print(f"PLMProfiler initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Disk cache: {use_disk_cache}")
        print(f"  RAM cache: {use_ram_cache}")
        print(f"  UniProt timeout: {uniprot_timeout}s")
    
    def _extract_window_positions(self, center_position: int, window_size: int) -> List[int]:
        """
        Extract positions for each residue in the window
        
        Args:
            center_position: Center position of the window (1-based)
            window_size: Size of the window
            
        Returns:
            List of positions in the window (only valid positions within sequence), 1-based.
            e.g. center_position = 10, window_size = 20, return [1, 2, 3, ..., 20]
        """
        half_window = window_size // 2
        start_pos = center_position - half_window + (1 if window_size % 2 == 0 else 0) # 1-based
        end_pos = center_position + half_window # 1-based
        
        # Only include positions within the sequence bounds
        positions = []
        for pos in range(start_pos, end_pos + 1):
            positions.append(pos)
        
        return positions # 1-based
    
    
    def _get_sequence_for_uniprot(self, uniprot_id: str) -> str:
        """
        Get the protein sequence for a given UniProt ID with caching support
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            str: Protein sequence
        """
        import requests
        
        # Generate cache key
        cache_key = self._generate_cache_key(uniprot_id=uniprot_id, data_type="sequence")
        
        # Try to load from cache first
        cached_sequence = self._get_cached_data(cache_key, 'sequence')
        if cached_sequence is not None:
            print(f" Loaded sequence for {uniprot_id} from cache (length: {len(cached_sequence)})")
            return cached_sequence
        
        # Fetch from UniProt API
        print(f"Fetching sequence for {uniprot_id} from UniProt...")
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        
        try:
            response = requests.get(url, timeout=self.uniprot_timeout)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])  # skip the FASTA header
                
                # Save to cache
                self._cache_data(cache_key, sequence, 'sequence')
                print(f" Downloaded and cached sequence for {uniprot_id} (length: {len(sequence)})")
                
                return sequence
            else:
                raise ValueError(f"Cannot get the sequence for UniProt ID {uniprot_id}, HTTP status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise ValueError(f"Timeout while downloading sequence for UniProt ID {uniprot_id} (timeout: {self.uniprot_timeout}s)")
        except requests.exceptions.ConnectionError as e:
            raise ValueError(f"Connection error while downloading sequence for UniProt ID {uniprot_id}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error while downloading sequence for UniProt ID {uniprot_id}: {str(e)}")


class ProstT5Profiler(PLMProfiler):
    """
    Window profiler that integrates WindowSlicer functionality for sequence windowing
    """
    
    def __init__(self, 
                 cache_dir: str = "structure_temp/prostt5_cache",
                 use_disk_cache: bool = True, 
                 use_ram_cache: bool = True,
                 uniprot_timeout: int = 30):
        """
        Initialize Window Profiler
        
        Args:
            cache_dir: Directory for cache storage
            use_disk_cache: Enable disk caching
            use_ram_cache: Enable RAM caching
            uniprot_timeout: Timeout for UniProt API requests (seconds)
        """
        super().__init__(cache_dir, use_disk_cache, use_ram_cache)
        
    def profile_plm(self, windows_df: pd.DataFrame,
                     embedding_level: str = 'residue', 
                     show_original_data: bool = True):
        """
        Profile PLM features for all windows
        
        Args:
            windows_df: DataFrame containing window information
            embedding_level: 'residue' for per-residue embeddings, 'window' for aggregated window embeddings
            show_original_data: Whether to include original window data in results
            
        Returns:
            DataFrame with PLM embeddings added
        """
        prostt5 = ProstT5()
        
        # Initialize results storage
        results = []
        
        # Process each window
        for index, row in windows_df.iterrows():
            uniprot_id = row['entry']
            center_position = row['center_position']
            window_size = row['window_size']
            
            try:
                # Get sequence and full protein embeddings (ProstT5 handles caching internally)
                sequence = self._get_sequence_for_uniprot(uniprot_id)
                print(f"Processing {uniprot_id}, sequence length: {len(sequence)}")
                
                # Get full sequence embeddings - this method handles its own caching
                _, residue_embeddings = prostt5.get_full_sequence_embeddings(sequence)
                
                # residue_embeddings is a list with one tensor of shape (seq_len, 1024)
                residue_tensor = residue_embeddings[0]  # Extract tensor from list
                print(f"Residue embeddings shape: {residue_tensor.shape}")
                
                # Extract window positions
                positions = self._extract_window_positions(center_position, window_size)
                
                # Convert positions to 0-based indexing for tensor indexing
                positions_0based = [pos - 1 for pos in positions]
                
                # Extract window embeddings
                window_embeddings = residue_tensor[positions_0based]  # Shape: (window_size, 1024)
                
                if embedding_level == 'residue':
                    # Return per-residue embeddings
                    window_result = {
                        'plm_embeddings': window_embeddings.detach().cpu().numpy()
                    }
                    
                elif embedding_level == 'window':
                    # Return aggregated window embedding
                    window_embedding = window_embeddings.mean(dim=0)  # Average across positions
                    window_result = {
                        'plm_embedding': window_embedding.detach().cpu().numpy()
                    }
                else:
                    raise ValueError(f"Invalid embedding level: {embedding_level}")
                
                results.append(window_result)
                    
            except Exception as e:
                print(f"Error processing {uniprot_id}: {str(e)}")
                # Create empty result
                if embedding_level == 'residue':
                    window_result = {'plm_embeddings': np.zeros((window_size, 1024))}
                else:
                    window_result = {'plm_embedding': np.zeros(1024)}
                results.append(window_result)
        
        # Convert results to DataFrame
        if embedding_level == 'residue':
            # For residue level, create columns pos_1, pos_2, etc., each storing a 1024-dim vector
            embedding_data = []
            for result in results:
                embeddings = result['plm_embeddings']  # Shape: (window_size, 1024)
                row_data = {}
                for pos_idx in range(embeddings.shape[0]):
                    col_name = f'pos_{pos_idx + 1}'  # pos_1, pos_2, etc.
                    row_data[col_name] = embeddings[pos_idx]  # Store 1024-dim vector
                embedding_data.append(row_data)
        else:
            # For window level, create single column prostt5_embedding storing the 1024-dim vector
            embedding_data = []
            for result in results:
                embedding = result['plm_embedding']  # Shape: (1024,)
                row_data = {'prostt5_embedding': embedding}
                embedding_data.append(row_data)
        
        # Create DataFrame from embedding data
        embeddings_df = pd.DataFrame(embedding_data, index=windows_df.index)
        
        # Combine with original data if requested
        if show_original_data:
            result_df = pd.concat([windows_df, embeddings_df], axis=1)
        else:
            result_df = embeddings_df
        
        print(f"\nPLM profiling completed. Added {len(embeddings_df.columns)} features.")
        print(f"Embedding level: {embedding_level}")
        
        return result_df

class ESM2Profiler(PLMProfiler):
    """
    ESM-2 Window profiler following the same pattern as ProstT5Profiler
    """
    
    def __init__(self, 
                 cache_dir: str = "structure_temp/esm2_cache",
                 use_disk_cache: bool = True, 
                 use_ram_cache: bool = True,
                 uniprot_timeout: int = 30):
        """
        Initialize ESM-2 Window Profiler
        
        Args:
            cache_dir: Directory for cache storage
            use_disk_cache: Enable disk caching
            use_ram_cache: Enable RAM caching
            uniprot_timeout: Timeout for UniProt API requests (seconds)
        """
        super().__init__(cache_dir, use_disk_cache, use_ram_cache, uniprot_timeout)
    
    def profile_plm(self, windows_df: pd.DataFrame,
                     embedding_level: str = 'residue', 
                     show_original_data: bool = True):
        """
        Profile ESM-2 features for all windows
        
        Args:
            windows_df: DataFrame containing window information
            embedding_level: 'residue' for per-residue embeddings, 'window' for aggregated window embeddings
            show_original_data: Whether to include original window data in results
            
        Returns:
            DataFrame with ESM-2 embeddings added
        """
        # Import ESM-2 here to handle optional dependency
        try:
            from cleavage_site_predictor.plm.esm2 import ESM2
        except ImportError as e:
            raise ImportError(
                "ESM-2 not available. Please install fair-esm in a Python 3.9 environment:\n"
                "conda create -n esm python=3.9\n"
                "conda activate esm\n"
                "pip install torch\n"
                "pip install fair-esm\n"
                f"Error: {e}"
            )
        
        esm2 = ESM2()
        
        # Initialize results storage
        results = []
        
        # Process each window
        for index, row in windows_df.iterrows():
            uniprot_id = row['entry']
            center_position = row['center_position']
            window_size = row['window_size']
            
            try:
                # Get sequence and full protein embeddings (ESM-2 handles caching internally)
                sequence = self._get_sequence_for_uniprot(uniprot_id)
                print(f"Processing {uniprot_id}, sequence length: {len(sequence)}")
                
                # Get full sequence embeddings - this method handles its own caching
                _, residue_embeddings = esm2.get_full_sequence_embeddings(sequence)
                
                # residue_embeddings is a list with one tensor of shape (seq_len, 1280)
                residue_tensor = residue_embeddings[0]  # Extract tensor from list
                print(f"ESM-2 embeddings shape: {residue_tensor.shape}")
                
                # Extract window positions
                positions = self._extract_window_positions(center_position, window_size)
                
                # Convert positions to 0-based indexing for tensor indexing
                positions_0based = [pos - 1 for pos in positions]
                
                # Extract window embeddings
                window_embeddings = residue_tensor[positions_0based]  # Shape: (window_size, 1280)
                
                if embedding_level == 'residue':
                    # Return per-residue embeddings
                    window_result = {
                        'plm_embeddings': window_embeddings.detach().cpu().numpy()
                    }
                    
                elif embedding_level == 'window':
                    # Return aggregated window embedding
                    window_embedding = window_embeddings.mean(dim=0)  # Average across positions
                    window_result = {
                        'plm_embedding': window_embedding.detach().cpu().numpy()
                    }
                else:
                    raise ValueError(f"Invalid embedding level: {embedding_level}")
                
                results.append(window_result)
                    
            except Exception as e:
                print(f"Error processing {uniprot_id}: {str(e)}")
                # Create empty result
                if embedding_level == 'residue':
                    window_result = {'plm_embeddings': np.zeros((window_size, 1280))}
                else:
                    window_result = {'plm_embedding': np.zeros(1280)}
                results.append(window_result)
        
        # Convert results to DataFrame
        if embedding_level == 'residue':
            # For residue level, create columns pos_1, pos_2, etc., each storing a 1280-dim vector
            embedding_data = []
            for result in results:
                embeddings = result['plm_embeddings']  # Shape: (window_size, 1280)
                row_data = {}
                for pos_idx in range(embeddings.shape[0]):
                    col_name = f'pos_{pos_idx + 1}'  # pos_1, pos_2, etc.
                    row_data[col_name] = embeddings[pos_idx]  # Store 1280-dim vector
                embedding_data.append(row_data)
        else:
            # For window level, create single column esm2_embedding storing the 1280-dim vector
            embedding_data = []
            for result in results:
                embedding = result['plm_embedding']  # Shape: (1280,)
                row_data = {'esm2_embedding': embedding}
                embedding_data.append(row_data)
        
        # Create DataFrame from embedding data
        embeddings_df = pd.DataFrame(embedding_data, index=windows_df.index)
        
        # Combine with original data if requested
        if show_original_data:
            result_df = pd.concat([windows_df, embeddings_df], axis=1)
        else:
            result_df = embeddings_df
        
        print(f"\nESM-2 profiling completed. Added {len(embeddings_df.columns)} features.")
        print(f"Embedding level: {embedding_level}")
        print(f"Embedding dimension: 1280 (ESM-2)")
        
        return result_df
        
            