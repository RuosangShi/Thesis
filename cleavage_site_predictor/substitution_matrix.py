'''
Substitution Matrix Profile
Profile the substitution matrix of the given sequence: PAM, BLOSUM, etc.
'''

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ._load_matrix import LoadMatrix

class SubstitutionMatrixProfile:
    """
    Substitution Matrix Profile (SMP)
    Uses substitution matrices (PAM, BLOSUM) to extract features for each residue in protein sequences
    Each residue gets features from all available substitution matrices
    """
    
    def __init__(self):
        """
        Initialize SubstitutionMatrixProfile
        """
        # Load all substitution matrices
        self.matrix_loader = LoadMatrix()
        
        # Define matrix names in order
        self.matrix_names = [
            'PAM30', 'PAM120', 'PAM300', 'PAM400', 'PAM500',
            'blosum30', 'blosum45', 'blosum62', 'blosum75', 'blosum100'
        ]
        
        # Load all matrices
        self.matrices = {}
        for matrix_name in self.matrix_names:
            if matrix_name.startswith('PAM'):
                self.matrices[matrix_name] = self.matrix_loader.PAMDict[matrix_name]
            else:
                self.matrices[matrix_name] = self.matrix_loader.blosumDict[matrix_name]
        
        # 20 standard amino acids
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # Create feature names
        self.feature_names = []
        for matrix_name in self.matrix_names:
            self.feature_names.append(f'SMP_{matrix_name}')
        
        print(f"SubstitutionMatrixProfile initialized with {len(self.matrix_names)} matrices")
    
    def get_residue_features(self, residue: str) -> List[float]:
        """
        Extract features for a single residue from all substitution matrices
        
        Parameters:
        residue: str, single amino acid residue
        
        Returns:
        List[float]: feature vector for the residue (10 dimensions)
        """
        features = []
        
        # Handle unknown amino acids
        if residue not in self.amino_acids:
            # Return zeros for unknown residues
            return [0.0] * len(self.matrix_names)
        
        # Get the amino acid index for matrix lookup
        aa_index = self.amino_acids.index(residue)
        
        # Extract diagonal value (self-substitution score) from each matrix
        for matrix_name in self.matrix_names:
            matrix = self.matrices[matrix_name]
            # Get self-substitution score (diagonal value)
            score = matrix[residue][aa_index]
            features.append(float(score))
        
        return features
    
    def calculate_sequence_features(self, sequence: str) -> np.ndarray:
        """
        Calculate substitution matrix features for a sequence
        
        Parameters:
        sequence: str, protein sequence
        
        Returns:
        np.ndarray: feature matrix (sequence_length x num_matrices)
        """
        sequence_features = []
        
        for residue in sequence:
            residue_features = self.get_residue_features(residue)
            sequence_features.append(residue_features)
        
        return np.array(sequence_features)
    
    def calculate_window_features(self, sequence: str, aggregation_method: str = 'mean') -> np.ndarray:
        """
        Calculate aggregated substitution matrix features for a window sequence
        
        Parameters:
        sequence: str, protein sequence (window)
        aggregation_method: str, method to aggregate features ('mean', 'sum', 'max', 'min')
        
        Returns:
        np.ndarray: aggregated feature vector (num_matrices dimensions)
        """
        sequence_features = self.calculate_sequence_features(sequence)
        
        if aggregation_method == 'mean':
            return np.mean(sequence_features, axis=0)
        elif aggregation_method == 'sum':
            return np.sum(sequence_features, axis=0)
        elif aggregation_method == 'max':
            return np.max(sequence_features, axis=0)
        elif aggregation_method == 'min':
            return np.min(sequence_features, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def transform(self, df_window: pd.DataFrame, mode: str = 'window', show_flattened_features: bool = True,
                  aggregation_method: str = 'mean', show_original_data: bool = True) -> pd.DataFrame:
        """
        Calculate substitution matrix features for sequences in DataFrame
        
        Parameters:
        df_window: pd.DataFrame, DataFrame containing sequence information
                   Must contain 'sequence' column
        mode: str, feature extraction mode ('window' or 'residue')
              'window': aggregate features for each sequence window
              'residue': features for each residue position
        aggregation_method: str, aggregation method for 'window' mode
        show_original_data: bool, whether to include original data in result
        
        Returns:
        pd.DataFrame: DataFrame containing substitution matrix features
        """
        if 'sequence' not in df_window.columns:
            raise ValueError("Input DataFrame must contain 'sequence' column")
        
        if mode == 'window':
            # Calculate aggregated features for each window
            smp_features = []
            for sequence in df_window['sequence']:
                features = self.calculate_window_features(sequence, aggregation_method)
                smp_features.append(features)
            
            # Convert to DataFrame
            feature_names = [f'{name}_{aggregation_method}' for name in self.feature_names]
            smp_df = pd.DataFrame(smp_features, columns=feature_names)
            
        elif mode == 'residue':
            # Calculate features for each residue position
            smp_features = []
            max_length = max(len(seq) for seq in df_window['sequence'])
            
            for sequence in df_window['sequence']:
                seq_features = self.calculate_sequence_features(sequence)
                
                # Pad sequences to the same length if needed
                if len(seq_features) < max_length:
                    padding = np.zeros((max_length - len(seq_features), len(self.matrix_names)))
                    seq_features = np.vstack([seq_features, padding])
                
                smp_features.append(seq_features)
            
            if show_flattened_features:
                # Flatten features for all positions
                all_features = []
                feature_names = []
                for pos in range(max_length):
                    for matrix_name in self.feature_names:
                        feature_names.append(f'{matrix_name}_pos{pos+1}')
                    
                for seq_features in smp_features:
                    flattened = seq_features.flatten()
                    all_features.append(flattened)
                    
                smp_df = pd.DataFrame(all_features, columns=feature_names)
            else:
                processed_features = []
                for seq_features in smp_features:
                    # Convert each position's features to a list
                    row_data = [seq_features[i].tolist() for i in range(len(seq_features))]
                    processed_features.append(row_data)
                
                feature_names = [f'pos_{i+1}' for i in range(max_length)]
                smp_df = pd.DataFrame(processed_features, columns=feature_names)
        
        if not show_original_data:
            return smp_df
        
        # Merge original data and features
        result_df = pd.concat([df_window.reset_index(drop=True), smp_df], axis=1)
        return result_df
    
    def analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Analyze a single sequence and return detailed substitution matrix information
        
        Parameters:
        sequence: str, protein sequence
        
        Returns:
        dict: dictionary containing substitution matrix analysis results
        """
        sequence_features = self.calculate_sequence_features(sequence)
        window_features = self.calculate_window_features(sequence, 'mean')
        
        # Calculate statistics for each matrix
        matrix_stats = {}
        for i, matrix_name in enumerate(self.matrix_names):
            matrix_values = sequence_features[:, i]
            matrix_stats[matrix_name] = {
                'mean': np.mean(matrix_values),
                'std': np.std(matrix_values),
                'min': np.min(matrix_values),
                'max': np.max(matrix_values),
                'median': np.median(matrix_values)
            }
        
        return {
            'sequence_length': len(sequence),
            'sequence_features_shape': sequence_features.shape,
            'window_features': dict(zip(self.feature_names, window_features)),
            'matrix_statistics': matrix_stats,
            'available_matrices': self.matrix_names
        }
    