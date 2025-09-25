'''composition of K-Spaced Amino Acid Pairs'''

import pandas as pd
import numpy as np
from itertools import product

class CKSAAP:
    """
    Composition of K-Spaced Amino Acid Pairs (CKSAAP)
    Used to extract sequential information features of amino acid pairs in protein sequences
    """
    
    def __init__(self, k_values=[0, 1, 2, 3]):
        """
        Initialize CKSAAP
        
        Args:
            k_values: list, K-space value list, default is [0, 1, 2, 3]
        """
        self.k_values = k_values
        # 20 standard amino acids
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # Generate all possible amino acid pairs (441: 20×20+1, including unknown amino acids)
        self.aa_pairs = [''.join(pair) for pair in product(self.amino_acids, repeat=2)]
        
        # Feature names
        self.feature_names = []
        for k in self.k_values:
            for pair in self.aa_pairs:
                self.feature_names.append(f'{pair}_{k}space')
    
    def extract_k_spaced_pairs(self, sequence, k):
        """
        Extract k-space amino acid pairs from sequence
        
        Parameters:
        sequence: str, protein sequence
        k: int, space value
        
        Returns:
        list: k-space amino acid pairs list
        """
        pairs = []
        for i in range(len(sequence) - k - 1):
            aa1 = sequence[i]
            aa2 = sequence[i + k + 1]
            # Only consider standard amino acids
            if aa1 in self.amino_acids and aa2 in self.amino_acids:
                pairs.append(aa1 + aa2)
        return pairs
    
    def calculate_cksaap_features(self, sequence):
        """
        Calculate CKSAAP features for a single sequence
        
        Parameters:
        sequence: str, protein sequence
        
        Returns:
        np.array: CKSAAP feature vector
        """
        features = []
        
        for k in self.k_values:
            # Extract k-space amino acid pairs
            k_pairs = self.extract_k_spaced_pairs(sequence, k)
            
            # Calculate frequency of each amino acid pair
            pair_counts = {}
            for pair in self.aa_pairs:
                pair_counts[pair] = k_pairs.count(pair)
            
            # Normalize (divide by total number of k-space pairs)
            total_pairs = len(k_pairs)
            if total_pairs > 0:
                pair_frequencies = [pair_counts[pair] / total_pairs for pair in self.aa_pairs]
            else:
                pair_frequencies = [0.0] * len(self.aa_pairs)
            
            features.extend(pair_frequencies)
        
        return np.array(features)
    
    def transform(self, df_window, show_original_data=True):
        """
        Calculate CKSAAP features for sequences in DataFrame
        
        Parameters:
        df_window: pd.DataFrame, DataFrame containing sequence information
                   Must contain 'sequence' column
        
        Returns:
        pd.DataFrame: DataFrame containing CKSAAP features
        """
        if 'sequence' not in df_window.columns:
            raise ValueError("Input DataFrame must contain 'sequence' column")
        
        # Calculate CKSAAP features for each sequence
        cksaap_features = []
        for sequence in df_window['sequence']:
            features = self.calculate_cksaap_features(sequence)
            cksaap_features.append(features)
        
        # Convert to DataFrame
        cksaap_df = pd.DataFrame(cksaap_features, columns=self.feature_names)
        if not show_original_data:
            return cksaap_df
        
        # Merge original data and features
        result_df = pd.concat([df_window.reset_index(drop=True), cksaap_df], axis=1)
        return result_df