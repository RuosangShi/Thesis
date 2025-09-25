'''coevolutionary patterns'''

import pandas as pd
import numpy as np
import math
import re
import warnings
warnings.filterwarnings("ignore")

class coevo_patterns:
    """
    Variable length coevolutionary patterns extraction class
    Based on EV-HIV methodology with A_A, A_AB, and AB_A pattern types
    """
    
    def __init__(self, confidence_level=0.95, critical_value=1.96, min_support=2, select_top_features=True, max_features=100):
        """
        Initialize coevolutionary patterns extractor
        
        Args:
            confidence_level: float, confidence level for pattern significance (default: 0.95)
            min_support: int, minimum support count for patterns (default: 2)
            critical_value: float, critical value for significance testing (default: 1.96)
            select_top_features: bool, whether to select top features (default: True)
            max_features: int, maximum number of features to select (default: 100)
        """
        self.confidence_level = confidence_level
        self.min_support = min_support
        # Critical value for 95% confidence level (1.96 for normal distribution)
        self.critical_value = critical_value
        self.select_top_features = select_top_features
        self.max_features = max_features
        
        # 20 standard amino acids
        self.amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        
        # Generate all possible amino acid pairs (400 combinations)
        self.amino_acids_2 = []
        for i in range(len(self.amino_acids)):
            for j in range(len(self.amino_acids)):
                self.amino_acids_2.append(self.amino_acids[i] + self.amino_acids[j])
    
    def _extract_positive_sequences(self, windows_df):
        """
        Extract positive sequences (cleavage sites) for pattern mining
        
        Args:
            windows_df: DataFrame containing sequences and labels
            
        Returns:
            list: positive sequences
        """
        if 'known_cleavage_site' in windows_df.columns:
            positive_mask = windows_df['known_cleavage_site'] == 1
            positive_sequences = windows_df[positive_mask]['sequence'].tolist()
        else:
            # If no labels, use all sequences
            positive_sequences = windows_df['sequence'].tolist()
        
        print(f"Using {len(positive_sequences)} positive sequences for pattern extraction")
        return positive_sequences
    
    def _weight_calculation(self, p_ij, p_i_, p__j):
        """
        Calculate weight using the EV-HIV formula
        
        Args:
            p_ij: joint probability
            p_i_: marginal probability of first element
            p__j: marginal probability of second element
            
        Returns:
            float: weight value
        """
        if p_ij == p_i_ or p_ij <= 0 or p_i_ <= 0 or p__j <= 0:
            return 0
        
        try:
            if (p_i_ - p_ij) <= 0 or (1 - p__j) <= 0:
                return 0
            
            w = np.log(p_ij / (p_i_ * p__j)) - np.log((p_i_ - p_ij) / (p_i_ * (1 - p__j)))
            return round(w, 2) if not (math.isnan(w) or math.isinf(w)) else 0
            
        except (ValueError, ZeroDivisionError):
            return 0
    
    def _extract_A_A_patterns(self, sequences):
        """
        Extract A_A patterns (single amino acid to single amino acid)
        """
        patterns = []
        
        for n in range(7):  # n from 0 to 6
            # Create occurrence matrix (20x20)
            M = [[0]*20 for _ in range(20)]
            
            for i in range(20):
                for j in range(i, 20):  # Only upper triangle to avoid duplication
                    p1 = self.amino_acids[i] + '.' * n + self.amino_acids[j]
                    p2 = self.amino_acids[j] + '.' * n + self.amino_acids[i]
                    
                    for seq in sequences:
                        if len(re.findall(p1, seq)):
                            M[i][j] += 1
                        if i != j and len(re.findall(p2, seq)):
                            M[j][i] += 1
            
            # Calculate probabilities and diff scores
            row_sum = [sum(row) for row in M]
            col_sum = [sum(M[i][j] for i in range(20)) for j in range(20)]
            total_sum = sum(row_sum)
            
            if total_sum == 0:
                continue
            
            for i in range(20):
                for j in range(20):
                    if M[i][j] < self.min_support:
                        continue
                    
                    p_ij = M[i][j] / total_sum
                    p_i_ = row_sum[i] / total_sum
                    p__j = col_sum[j] / total_sum
                    
                    if p_i_ == 0 or p__j == 0:
                        continue
                    
                    # Calculate diff score
                    try:
                        diff = (p_ij - (p_i_ * p__j)) / np.sqrt((p_i_ * p__j * (1 - p_i_) * (1 - p__j)) / total_sum)
                        
                        if diff >= self.critical_value:
                            w = self._weight_calculation(p_ij, p_i_, p__j)
                            if w > 0:
                                pattern_str = self.amino_acids[i] + '.' * n + self.amino_acids[j]
                                patterns.append({
                                    'pattern': pattern_str,
                                    'weight': w,
                                    'count': M[i][j],
                                    'type': 'A_A',
                                    'diff': diff
                                })
                    except (ValueError, ZeroDivisionError):
                        continue
        
        return patterns
    
    def _extract_A_AB_patterns(self, sequences):
        """
        Extract A_AB patterns (single amino acid to double amino acid)
        """
        patterns = []
        
        for n in range(6):  # n from 0 to 5
            # Create occurrence matrix (20x400)
            M = [[0]*400 for _ in range(20)]
            
            for i in range(20):
                for j in range(400):
                    p = self.amino_acids[i] + '.' * n + self.amino_acids_2[j]
                    
                    for seq in sequences:
                        if len(re.findall(p, seq)):
                            M[i][j] += 1
            
            # Calculate probabilities and diff scores
            row_sum = [sum(row) for row in M]
            col_sum = [sum(M[i][j] for i in range(20)) for j in range(400)]
            total_sum = sum(row_sum)
            
            if total_sum == 0:
                continue
            
            for i in range(20):
                for j in range(400):
                    if M[i][j] < self.min_support:
                        continue
                    
                    p_ij = M[i][j] / total_sum
                    p_i_ = row_sum[i] / total_sum
                    p__j = col_sum[j] / total_sum
                    
                    if p_i_ == 0 or p__j == 0:
                        continue
                    
                    # Calculate diff score
                    try:
                        diff = (p_ij - (p_i_ * p__j)) / np.sqrt((p_i_ * p__j * (1 - p_i_) * (1 - p__j)) / total_sum)
                        
                        if diff >= self.critical_value:
                            w = self._weight_calculation(p_ij, p_i_, p__j)
                            if w > 0:
                                pattern_str = self.amino_acids[i] + '.' * n + self.amino_acids_2[j]
                                patterns.append({
                                    'pattern': pattern_str,
                                    'weight': w,
                                    'count': M[i][j],
                                    'type': 'A_AB',
                                    'diff': diff
                                })
                    except (ValueError, ZeroDivisionError):
                        continue
        
        return patterns
    
    def _extract_AB_A_patterns(self, sequences):
        """
        Extract AB_A patterns (double amino acid to single amino acid)
        """
        patterns = []
        
        for n in range(6):  # n from 0 to 5
            # Create occurrence matrix (400x20)
            M = [[0]*20 for _ in range(400)]
            
            for i in range(400):
                for j in range(20):
                    p = self.amino_acids_2[i] + '.' * n + self.amino_acids[j]
                    
                    for seq in sequences:
                        if len(re.findall(p, seq)):
                            M[i][j] += 1
            
            # Calculate probabilities and diff scores
            row_sum = [sum(row) for row in M]
            col_sum = [sum(M[i][j] for i in range(400)) for j in range(20)]
            total_sum = sum(row_sum)
            
            if total_sum == 0:
                continue
            
            for i in range(400):
                for j in range(20):
                    if M[i][j] < self.min_support:
                        continue
                    
                    p_ij = M[i][j] / total_sum
                    p_i_ = row_sum[i] / total_sum
                    p__j = col_sum[j] / total_sum
                    
                    if p_i_ == 0 or p__j == 0:
                        continue
                    
                    # Calculate diff score
                    try:
                        diff = (p_ij - (p_i_ * p__j)) / np.sqrt((p_i_ * p__j * (1 - p_i_) * (1 - p__j)) / total_sum)
                        
                        if diff >= self.critical_value:
                            w = self._weight_calculation(p_ij, p_i_, p__j)
                            if w > 0:
                                pattern_str = self.amino_acids_2[i] + '.' * n + self.amino_acids[j]
                                patterns.append({
                                    'pattern': pattern_str,
                                    'weight': w,
                                    'count': M[i][j],
                                    'type': 'AB_A',
                                    'diff': diff
                                })
                    except (ValueError, ZeroDivisionError):
                        continue
        
        return patterns
    
    def extract_coevolutionary_patterns(self, windows_df):
        """
        Extract all types of coevolutionary patterns from windows DataFrame
        
        Args:
            windows_df: pd.DataFrame containing sequence windows with 'sequence' column
            
        Returns:
            pd.DataFrame: DataFrame with coevolutionary pattern features
        """
        if 'sequence' not in windows_df.columns:
            raise ValueError("Input DataFrame must contain 'sequence' column")
        
        # Extract positive sequences for pattern mining
        positive_sequences = self._extract_positive_sequences(windows_df)
        
        if not positive_sequences:
            print("No positive sequences found, using all sequences")
            positive_sequences = windows_df['sequence'].tolist()
        
        print("Extracting coevolutionary patterns...")
        
        # Extract all three types of patterns
        print("Extracting A_A patterns...")
        aa_patterns = self._extract_A_A_patterns(positive_sequences)
        
        print("Extracting A_AB patterns...")
        a_ab_patterns = self._extract_A_AB_patterns(positive_sequences)
        
        print("Extracting AB_A patterns...")
        ab_a_patterns = self._extract_AB_A_patterns(positive_sequences)
        
        # Combine all patterns
        all_patterns = aa_patterns + a_ab_patterns + ab_a_patterns
        
        print(f"Found {len(all_patterns)} significant patterns:")
        print(f"  A_A patterns: {len(aa_patterns)}")
        print(f"  A_AB patterns: {len(a_ab_patterns)}")
        print(f"  AB_A patterns: {len(ab_a_patterns)}")
        
        # Sort patterns by diff score in descending order and select top features
        all_patterns_sorted = sorted(all_patterns, key=lambda x: x['diff'], reverse=True)
        
        # Select top max_features patterns
        if self.select_top_features and len(all_patterns_sorted) > self.max_features:
            selected_patterns = all_patterns_sorted[:self.max_features]
            print(f"Selected top {self.max_features} patterns based on diff score")
        else:
            selected_patterns = all_patterns_sorted
            print(f"Using all {len(selected_patterns)} patterns (less than max_features={self.max_features})")
        
        # Update pattern counts after selection
        selected_aa = [p for p in selected_patterns if p['type'] == 'A_A']
        selected_a_ab = [p for p in selected_patterns if p['type'] == 'A_AB']
        selected_ab_a = [p for p in selected_patterns if p['type'] == 'AB_A']
        
        print(f"Selected patterns by type:")
        print(f"  A_A patterns: {len(selected_aa)}")
        print(f"  A_AB patterns: {len(selected_a_ab)}")
        print(f"  AB_A patterns: {len(selected_ab_a)}")
        
        if selected_patterns:
            print(f"Diff score range: {selected_patterns[0]['diff']:.3f} to {selected_patterns[-1]['diff']:.3f}")
        
        self.significant_patterns = selected_patterns
        self.all_significant_patterns = all_patterns
        return selected_patterns, all_patterns

    
    def profile_windows(self, windows_df, sequence_col='sequence', selected_patterns=None, show_original_data=True):
        '''Use the given patterns to profile the windows
        
        Args:
            windows_df: pd.DataFrame containing sequence windows with 'sequence' column
            sequence_col: str, name of the column containing sequences
            selected_patterns: list, list of selected patterns
            show_original_data: bool, whether to show original data

        Returns:
            pd.DataFrame: DataFrame with coevolutionary pattern features
        '''
        if selected_patterns is None:
            selected_patterns = self.significant_patterns
        
        # Extract features for each sequence window
        coevo_features = []
        
        for sequence in windows_df[sequence_col]:
            seq_features = {}
            
            # Initialize all pattern features to 0 (using selected patterns only)
            for pattern in selected_patterns:
                pattern_name = f"coevo_{pattern['type']}_{pattern['pattern']}"
                seq_features[pattern_name] = 0.0
            
            # Count occurrences of each pattern in this sequence
            for pattern in selected_patterns:
                pattern_regex = pattern['pattern']
                weight = pattern['weight']
                pattern_name = f"coevo_{pattern['type']}_{pattern['pattern']}"
                
                # Count matches using regex
                matches = len(re.findall(pattern_regex, sequence))
                seq_features[pattern_name] = matches * weight
            
            coevo_features.append(seq_features)
        
        coevo_df = pd.DataFrame(coevo_features)
        if not show_original_data:
            return coevo_df
        result_df = pd.concat([windows_df, coevo_df], axis=1)
        return result_df

    