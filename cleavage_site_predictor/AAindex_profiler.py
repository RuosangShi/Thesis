'''This script is used to profile the AAindex of the cleavage site'''

import pandas as pd
import aaanalysis as aa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class AAindex_profiler:
    def __init__(self, reduce_scales: bool = True, n_clusters: int = 133):
        """
        Initialize the AAindex profiler
        
        Parameters:
        -----------
        reduce_scales : bool, default=True
            if True, reduce the number of scales
        n_clusters : int, default=133
            the number of clusters when reduce_scales=True
        """
        self.reduce_scales = reduce_scales
        self.n_clusters = n_clusters
        self.df_scales = None
    
    def get_aa_index(self) -> pd.DataFrame:
        """
        Get the AAindex scales
        
        Returns:
        --------
        pd.DataFrame
            the AAindex scales dataframe
        """
        # Load scales
        df_scales = aa.load_scales() # 586 scales in total
        if self.reduce_scales:
            # Obtain redundancy-reduced set of scales
            aac = aa.AAclust()
            X = np.array(df_scales).T
            scales = aac.fit(X, names=list(df_scales), n_clusters=self.n_clusters).medoid_names_
            df_scales = df_scales[scales]
        
        self.df_scales = df_scales
        print(f"Using {len(df_scales.columns)} AAindex scales")
        return df_scales
    
    def map_aa_index_to_windows(self, windows: pd.DataFrame, 
                               sequence_col: str = 'sequence',
                               show_flattened_features: bool = True) -> pd.DataFrame:
        """
        Map the AAindex to the sequence windows
        
        Parameters:
        -----------
        windows : pd.DataFrame
            the dataframe containing the sequence windows
        sequence_col : str, default='sequence'
            the column name of the sequence
            
        Returns:
        --------
        pd.DataFrame
            the mapped feature matrix
        """
        if self.df_scales is None:
            self.df_scales = self.get_aa_index()
        
        if sequence_col not in windows.columns:
            raise ValueError(f"The column '{sequence_col}' does not exist in the dataframe")
        
        print(f"Processing {len(windows)} sequence windows...")
        
        # Prepare data for SequenceFeature, using the column name format required by aaanalysis
        sequences = windows[sequence_col].tolist()
        
        # Create the feature matrix another way: directly calculate the AAindex value for each position
        sequence_length = len(sequences[0])
        n_sequences = len(sequences)
        n_scales = len(self.df_scales.columns)
        
        if show_flattened_features:
            # Initialize the feature matrix
            feature_matrix = np.zeros((n_sequences, sequence_length * n_scales))
            feature_names = []
            
            # Generate features for each position and each scale
            for pos in range(sequence_length):
                for scale_idx, scale_name in enumerate(self.df_scales.columns):
                    feature_names.append(f"pos_{pos+1}_{scale_name}")
                    
                    # Calculate the value of the scale at this position
                    for seq_idx, seq in enumerate(sequences):
                        if pos < len(seq) and seq[pos] in self.df_scales.index:
                            aa = seq[pos]
                            if aa != 'X':  # Skip the padding character
                                feature_matrix[seq_idx, pos * n_scales + scale_idx] = self.df_scales.loc[aa, scale_name]
        
            # Convert to DataFrame
            feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
        
        else:
            # Create non-flattened features - each position contains a vector of scale values
            all_features = []
            for seq in sequences:
                seq_features = []
                for pos in range(sequence_length):
                    if pos < len(seq) and seq[pos] in self.df_scales.index:
                        aa = seq[pos]
                        if aa != 'X':  # Skip padding character
                            pos_features = self.df_scales.loc[aa].values.tolist()
                        else:
                            pos_features = [0.0] * n_scales
                    else:
                        # Padding for positions beyond sequence length
                        pos_features = [0.0] * n_scales
                    seq_features.append(pos_features)
                all_features.append(seq_features)
            
            # Create DataFrame with position-based columns
            feature_names = [f'pos_{i+1}' for i in range(sequence_length)]
            feature_df = pd.DataFrame(all_features, columns=feature_names)            
        return feature_df
    
    def calculate_position_profiles(self, windows: pd.DataFrame, 
                                  label_col: str = 'known_cleavage_site',
                                  sequence_col: str = 'sequence') -> Dict:
        """
        Calculate the AAindex profile for each position
        
        Parameters:
        -----------
        windows : pd.DataFrame
            the dataframe containing the sequence windows and labels
        label_col : str, default='known_cleavage_site'
            the column name of the label (1 for cleavage site, 0 for non-cleavage site)
        sequence_col : str, default='sequence'
            the column name of the sequence
            
        Returns:
        --------
        Dict
            the dictionary containing the AAindex profile for each position
        """
        if self.df_scales is None:
            self.df_scales = self.get_aa_index()
        
        # Separate positive and negative samples - using numerical comparison
        positive_seqs = windows[windows[label_col] == 1][sequence_col].tolist()
        negative_seqs = windows[windows[label_col] == 0][sequence_col].tolist()
        
        print(f"Positive samples: {len(positive_seqs)}, Negative samples: {len(negative_seqs)}")
        
        sequence_length = len(positive_seqs[0]) if positive_seqs else len(negative_seqs[0])
        
        position_profiles = {}
        
        for pos in range(sequence_length):
            position_profiles[pos + 1] = {}
            
            # Get the amino acid at this position
            pos_aa_positive = [seq[pos] for seq in positive_seqs if pos < len(seq) and seq[pos] != 'X']
            pos_aa_negative = [seq[pos] for seq in negative_seqs if pos < len(seq) and seq[pos] != 'X']
            
            # Calculate the AAindex value at this position
            for scale_name in self.df_scales.columns:
                if pos_aa_positive:
                    positive_values = [self.df_scales.loc[aa, scale_name] for aa in pos_aa_positive 
                                     if aa in self.df_scales.index]
                    position_profiles[pos + 1][f'{scale_name}_positive'] = np.mean(positive_values) if positive_values else 0
                else:
                    position_profiles[pos + 1][f'{scale_name}_positive'] = 0
                
                if pos_aa_negative:
                    negative_values = [self.df_scales.loc[aa, scale_name] for aa in pos_aa_negative 
                                     if aa in self.df_scales.index]
                    position_profiles[pos + 1][f'{scale_name}_negative'] = np.mean(negative_values) if negative_values else 0
                else:
                    position_profiles[pos + 1][f'{scale_name}_negative'] = 0
                
                # Calculate the difference
                position_profiles[pos + 1][f'{scale_name}_diff'] = (
                    position_profiles[pos + 1][f'{scale_name}_positive'] - 
                    position_profiles[pos + 1][f'{scale_name}_negative']
                )
        
        return position_profiles
    
    
    def run_complete_analysis(self, windows: pd.DataFrame,
                            sequence_col: str = 'sequence',
                            label_col: str = 'known_cleavage_site',
                            plot_results: bool = True,
                            top_scales: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete AAindex analysis
        
        Parameters:
        -----------
        windows : pd.DataFrame
            the dataframe containing the sequence windows
        sequence_col : str, default='sequence'
            the column name of the sequence
        label_col : str, default='known_cleavage_site'
            the column name of the label (1 for cleavage site, 0 for non-cleavage site)
        plot_results : bool, default=True
            if True, plot the results
        top_scales : int, default=5
            the number of top scales to plot
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            feature_matrix: the feature matrix
            position_profiles: the position profiles
        """
        print("=== Start AAindex analysis ===")
        
        # 1. Obtain the AAindex scales
        print("1. Obtain the AAindex scales...")
        self.get_aa_index()
        
        # 2. Map the AAindex to the sequence windows
        print("2. Map the AAindex to the sequence windows...")
        feature_matrix = self.map_aa_index_to_windows(windows, sequence_col)
        
        # 3. Calculate the position profiles
        print("3. Calculate the position profiles...")
        position_profiles = self.calculate_position_profiles(windows, label_col, sequence_col)
        
        # 4. Plot the results using new distribution plot
        if plot_results:
            print("4. Plot the results...")
            # Use the new distribution plot for top discriminative scales
            discriminative_scales = self.find_discriminative_features(position_profiles, top_n=top_scales)['scale'].unique()[:3]

            for scale_name in discriminative_scales:
                if scale_name in self.df_scales.columns:
                    self.plot_aaindex_distribution(
                        windows=windows,
                        labels=windows[label_col].values,
                        title=f"AAindex Distribution: {scale_name}",
                        scale_name=scale_name,
                        figsize=(10, 6)
                    )
        
        print("=== AAindex analysis completed ===")
        return feature_matrix, position_profiles
    