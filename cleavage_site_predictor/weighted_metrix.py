from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
from ._load_matrix import LoadMatrix


class BaseMatrix(LoadMatrix):
    """
    Base Matrix Class - Contains common utilities and visualization methods
    Each specific matrix type (PWM, PSSM, NNS, KNN, SMI) inherits from this class
    """
    
    def __init__(self, matrix_type: str = 'base', pseudocount: float = 0.01):
        """Initialize base matrix analyzer
        
        Args:
            matrix_type: Type of matrix
            pseudocount: Pseudocount for frequency calculations
        """
        super().__init__()  # Initialize LoadMatrix to get dictionaries
        self.matrix_type = matrix_type
        self.pseudocount = pseudocount
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.matrix = None
        self.frequency_matrix = None
        self.background_freq = None
        self.window_size = None
        self.training_sequences = None
        self.is_built = False
        
        print(f"{matrix_type.upper()} Matrix Analyzer initialized (Pseudocount: {pseudocount})")

    def _process_fasta_file(self, fasta_file: str) -> List[str]:
        """Process FASTA file and return sequence list"""
        with open(fasta_file, 'r') as file:
            sequences = []
            current_seq = ""
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
            return sequences
    
    def _calculate_background_frequency(self, sequences: List[str]) -> Dict[str, float]:
        """Calculate background amino acid frequencies"""
        aa_counts = defaultdict(int)
        total_count = 0
        
        for seq in sequences:
            for aa in seq:
                if aa in self.amino_acids:
                    aa_counts[aa] += 1
                    total_count += 1
        
        background_freq = {}
        for aa in self.amino_acids:
            background_freq[aa] = (aa_counts[aa] + self.pseudocount) / (total_count + len(self.amino_acids) * self.pseudocount)
        
        return background_freq
    
    def _plot_background_frequency(self, background_freq: Dict[str, float], 
                                 figsize: Tuple[int, int] = (12, 6),
                                 color_by_group: bool = True,
                                 title: str = None) -> Dict:
        """Analyze and plot background amino acid frequency
        
        Args:
            background_freq: Dictionary of amino acid frequencies
            figsize: Figure size as (width, height)
            color_by_group: Whether to color bars by amino acid groups
            title: Plot title (default: 'Background Amino Acid Frequency')
            
        Returns:
            Dictionary containing frequency analysis results
        """
        bg_freq_df = pd.DataFrame(list(background_freq.items()), 
                                 columns=['Amino Acid', 'Frequency'])
        bg_freq_df = bg_freq_df.sort_values('Frequency', ascending=False)
        bg_freq_df['Percentage'] = bg_freq_df['Frequency'] * 100
        
        print("Background Amino Acid Frequency:")
        for _, row in bg_freq_df.iterrows():
            print(f"  {row['Amino Acid']}: {row['Percentage']:.1f}%")
        
        plt.figure(figsize=figsize)
        
        # Define colors for different amino acid groups
        if color_by_group:
            aa_group_colors = {
                # Non-polar/Hydrophobic - blue shades
                'A': '#1f77b4', 'V': '#1f77b4', 'I': '#1f77b4', 'L': '#1f77b4', 
                'M': '#1f77b4', 'F': '#1f77b4', 'Y': '#1f77b4', 'W': '#1f77b4',
                # Polar uncharged - green shades
                'S': '#2ca02c', 'T': '#2ca02c', 'N': '#2ca02c', 'Q': '#2ca02c', 'C': '#2ca02c',
                # Positive charge - red shades
                'K': '#d62728', 'R': '#d62728', 'H': '#d62728',
                # Negative charge - orange shades
                'D': '#ff7f0e', 'E': '#ff7f0e',
                # Special - purple shades
                'G': '#9467bd', 'P': '#9467bd'
            }
            colors = [aa_group_colors.get(aa, 'skyblue') for aa in bg_freq_df['Amino Acid']]
        else:
            colors = 'skyblue'
        
        bars = plt.bar(bg_freq_df['Amino Acid'], bg_freq_df['Percentage'], 
                      color=colors, edgecolor='navy', alpha=0.7)
        
        # Set title
        if title is None:
            title = 'Background Amino Acid Frequency'
        plt.title(title, fontweight='bold', fontsize=14)
        plt.xlabel('Amino Acid', fontweight='bold', fontsize=12)
        plt.ylabel('Frequency (%)', fontweight='bold', fontsize=12)
        plt.xticks(rotation=45, fontweight='bold', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, bg_freq_df['Percentage']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        # Add legend if using color groups - position to avoid overlap
        if color_by_group:
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.7, label='Non-polar/Hydrophobic'),
                plt.Rectangle((0,0),1,1, facecolor='#2ca02c', alpha=0.7, label='Polar Uncharged'),
                plt.Rectangle((0,0),1,1, facecolor='#d62728', alpha=0.7, label='Positive Charge'),
                plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.7, label='Negative Charge'),
                plt.Rectangle((0,0),1,1, facecolor='#9467bd', alpha=0.7, label='Special')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=9,
                      frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for rotated x-axis labels
        plt.show()
        
        return {
            'frequency_data': bg_freq_df.to_dict('records'),
            'most_frequent': bg_freq_df.iloc[0]['Amino Acid'],
            'least_frequent': bg_freq_df.iloc[-1]['Amino Acid'],
            'frequency_range': [bg_freq_df['Frequency'].min(), bg_freq_df['Frequency'].max()],
            'top_5_amino_acids': bg_freq_df.head(5)['Amino Acid'].tolist(),
            'bottom_5_amino_acids': bg_freq_df.tail(5)['Amino Acid'].tolist()
        }
    
    def _calculate_position_frequency(self, positive_sequences: List[str]) -> np.ndarray:
        """Calculate position-specific frequency matrix"""
        if not positive_sequences:
            raise ValueError("No positive sequences provided")
        
        window_size = len(positive_sequences[0])
        freq_matrix = np.zeros((window_size, len(self.amino_acids)))
        
        for seq in positive_sequences:
            if len(seq) != window_size:
                continue
            
            for pos, aa in enumerate(seq):
                if aa in self.amino_acids:
                    aa_idx = self.amino_acids.index(aa)
                    freq_matrix[pos, aa_idx] += 1
        
        freq_matrix += self.pseudocount
        freq_matrix = freq_matrix / freq_matrix.sum(axis=1, keepdims=True)
        
        return freq_matrix

    # Abstract methods to be implemented by subclasses
    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site', **kwargs) -> None:
        """Build matrix from training data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build() method")
    
    def score_sequence(self, sequence: str) -> float:
        """Score a single sequence - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement score_sequence() method")
    
    def score_sequences(self, sequences: List[str]) -> List[float]:
        """Batch score sequences"""
        return [self.score_sequence(seq) for seq in sequences]
    
    def get_position_weights(self, position: int) -> Dict[str, float]:
        """Get amino acid weights for specific position"""
        if self.matrix is None:
            raise ValueError("Matrix not built")
        
        if position < 0 or position >= self.window_size:
            raise ValueError(f"Position index out of range: {position}")
        
        weights = {}
        for aa_idx, aa in enumerate(self.amino_acids):
            weights[aa] = self.matrix[position, aa_idx]
        
        return weights

    def get_matrix(self) -> pd.DataFrame:
        """Get matrix for PSSM, PWM, SMI"""
        if self.matrix is None:
            raise ValueError("Matrix not built")
        
        df_matrix = pd.DataFrame(self.matrix, columns=self.amino_acids,
                         index=[f'Position_{i+1}' for i in range(self.window_size)])
        return df_matrix

    def generate_score_dataframe(self, windows_df: pd.DataFrame,
                               sequence_col: str = 'sequence',
                               include_original_cols: bool = True) -> pd.DataFrame:
        """Generate a dataframe with scores from the matrix
        
        Args:
            windows_df: DataFrame containing sequences to score
            sequence_col: Column name containing sequences
            include_original_cols: Whether to include original columns in output
            
        Returns:
            DataFrame with added score columns
        """
        if not self.is_built:
            raise ValueError(f"{self.matrix_type.upper()} not built. Call build() first")
        
        valid_windows = windows_df[windows_df[sequence_col].str.len() == self.window_size].copy()
        
        print(f"Generating {self.matrix_type.upper()} scores for {len(valid_windows)} sequences...")
        
        # For SMI, handle multiple matrices
        if self.matrix_type.upper() == 'SMI':
            return self._generate_smi_score_dataframe(valid_windows, sequence_col, include_original_cols)
        
        # For other matrices, generate single score column
        score_col = f"{self.matrix_type.upper()}_SCORE"
        scores = []
        
        for seq in valid_windows[sequence_col]:
            score = self.score_sequence(seq)
            scores.append(score)
        
        valid_windows[score_col] = scores
        print(f"  ✅ {score_col}: done")
        
        print(f"\n🎉 {self.matrix_type.upper()} score dataframe generation completed!")
        print(f"   New columns: [{score_col}]")
        print(f"   Data shape: {valid_windows.shape}")
        
        if not include_original_cols:
            return valid_windows[[sequence_col, score_col]]
        
        return valid_windows


class PWM(BaseMatrix):
    """
    Position-Specific Weighted Matrix (PWM)
    Based on position-specific weighted matrix for sequence scoring
    """
    
    def __init__(self, pseudocount: float = 0.01):
        super().__init__('PWM', pseudocount)

    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site', 
                 self_defined_background_freq: bool = True,
              fasta_file: str = None,
              plot_background_frequency: bool = False) -> None:
        """Build PWM matrix from training data"""
        positive_mask = windows_df[label_col] == 1
        positive_sequences = windows_df[positive_mask][sequence_col].tolist()
        all_sequences = windows_df[sequence_col].tolist()
        
        if not positive_sequences:
            raise ValueError("No positive sample sequences found")
        
        self.window_size = len(positive_sequences[0])
        positive_sequences = [seq for seq in positive_sequences if len(seq) == self.window_size]
        
        print(f"Building PWM matrix: Window size={self.window_size}, Positive sample count={len(positive_sequences)}")
        
        # Calculate background frequency
        if self_defined_background_freq and fasta_file:
            self.background_freq = self._calculate_background_frequency(self._process_fasta_file(fasta_file))
        else:
            self.background_freq = self._calculate_background_frequency(all_sequences)
        
        if plot_background_frequency:
            self._plot_background_frequency(self.background_freq)
        
        # Calculate frequency matrix and log odds matrix
        self.frequency_matrix = self._calculate_position_frequency(positive_sequences)
        self.log_odds_matrix = np.zeros_like(self.frequency_matrix)
        
        for pos in range(self.window_size):
            for aa_idx, aa in enumerate(self.amino_acids):
                observed_freq = self.frequency_matrix[pos, aa_idx]
                background_freq = self.background_freq[aa]
                self.log_odds_matrix[pos, aa_idx] = np.log2(observed_freq / background_freq)
        
        self.matrix = self.log_odds_matrix
        self.is_built = True
        print("PWM matrix built successfully")

    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using PWM"""
        if not self.is_built:
            raise ValueError("PWM not built. Call build() first")
        
        if len(sequence) != self.window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) does not match window size ({self.window_size})")
        
        score = 0.0
        for pos, aa in enumerate(sequence):
            if aa in self.amino_acids:
                aa_idx = self.amino_acids.index(aa)
                score += self.matrix[pos, aa_idx]
        
        return score


class PSSM(BaseMatrix):
    """
    Position-Specific Scoring Matrix (PSSM)
    Based on BLOSUM62 substitution matrix for position-specific scoring
    """
    
    def __init__(self, pseudocount: float = 0.01):
        super().__init__('PSSM', pseudocount)
        self.blosum62 = self._load_blosum62()

    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site', 
              self_defined_background_freq: bool = True,
              fasta_file: str = None,
              plot_background_frequency: bool = False) -> None:
        """Build PSSM matrix from training data"""
        positive_mask = windows_df[label_col] == 1
        positive_sequences = windows_df[positive_mask][sequence_col].tolist()
        all_sequences = windows_df[sequence_col].tolist()
        
        if not positive_sequences:
            raise ValueError("No positive sample sequences found")
        
        self.window_size = len(positive_sequences[0])
        positive_sequences = [seq for seq in positive_sequences if len(seq) == self.window_size]
        
        print(f"Building PSSM matrix: Window size={self.window_size}, Positive sample count={len(positive_sequences)}")
        
        # Calculate background frequency
        if self_defined_background_freq and fasta_file:
            self.background_freq = self._calculate_background_frequency(self._process_fasta_file(fasta_file))
        else:
            self.background_freq = self._calculate_background_frequency(all_sequences)
        
        if plot_background_frequency:
            self._plot_background_frequency(self.background_freq)
        
        # Calculate frequency matrix
        self.frequency_matrix = self._calculate_position_frequency(positive_sequences)
        
        # Calculate score matrix according to PSSM formula
        self.matrix = np.zeros_like(self.frequency_matrix)
        
        for pos in range(self.window_size):
            for j_idx, aa_j in enumerate(self.amino_acids):
                pssm_score = 0.0
                
                for i_idx, aa_i in enumerate(self.amino_acids):
                    Pi = self.frequency_matrix[pos, i_idx]
                    blosum_score = self.blosum62[aa_i][j_idx]
                    pssm_score += Pi * (2 ** (blosum_score/2))
                
                self.matrix[pos, j_idx] = np.log2(pssm_score)
        
        self.is_built = True
        print("PSSM matrix built successfully")

    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using PSSM"""
        if not self.is_built:
            raise ValueError("PSSM not built. Call build() first")
        
        if len(sequence) != self.window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) does not match window size ({self.window_size})")
        
        score = 0.0
        for pos, aa in enumerate(sequence):
            if aa in self.amino_acids:
                aa_idx = self.amino_acids.index(aa)
                score += self.matrix[pos, aa_idx]
        
        return score


class NNS(BaseMatrix):
    """
    Nearest Neighbor Scoring (NNS)
    Compares query sequence against all training cleavage sequences using BLOSUM62
    """
    
    def __init__(self, pseudocount: float = 0.01):
        super().__init__('NNS', pseudocount)
        self.blosum62 = self._load_blosum62()

    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site') -> None:
        """Build NNS training set from cleavage windows"""
        positive_mask = windows_df[label_col] == 1
        positive_sequences = windows_df[positive_mask][sequence_col].tolist()
        
        if not positive_sequences:
            raise ValueError("No positive sample sequences found")
        
        self.window_size = len(positive_sequences[0])
        self.training_sequences = [seq for seq in positive_sequences if len(seq) == self.window_size]
        self.is_built = True
        print(f"Built NNS training set: Window size={self.window_size}, Training sequences={len(self.training_sequences)}")

    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using NNS"""
        if not self.is_built:
            raise ValueError("NNS not built. Call build() first")
        
        if len(sequence) != self.window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) does not match window size ({self.window_size})")
        
        total_score = 0.0
        for train_seq in self.training_sequences:
            for pos in range(self.window_size):
                q_aa = sequence[pos]
                t_aa = train_seq[pos]
                
                if q_aa in self.amino_acids and t_aa in self.amino_acids:
                    q_idx = self.amino_acids.index(q_aa)
                    t_idx = self.amino_acids.index(t_aa)
                    total_score += self.blosum62[q_aa][t_idx]
                    
        return total_score / (len(self.training_sequences) * self.window_size)


class KNN(BaseMatrix):
    """
    K-Nearest Neighbours (KNN)
    Identifies k most similar sequences and calculates average similarity
    """
    
    def __init__(self, k: int = 5, pseudocount: float = 0.01):
        super().__init__('KNN', pseudocount)
        self.k = k
        self.blosum62 = self._load_blosum62()

    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site') -> None:
        """Build KNN training set from cleavage windows"""
        positive_mask = windows_df[label_col] == 1
        positive_sequences = windows_df[positive_mask][sequence_col].tolist()
        
        if not positive_sequences:
            raise ValueError("No positive sample sequences found")
        
        self.window_size = len(positive_sequences[0])
        self.training_sequences = [seq for seq in positive_sequences if len(seq) == self.window_size]
        self.is_built = True
        print(f"Built KNN training set: Window size={self.window_size}, Training sequences={len(self.training_sequences)}, K={self.k}")

    def _calculate_similarity(self, aa1: str, aa2: str) -> float:
        """Calculate normalized similarity between two amino acids using BLOSUM62"""
        if aa1 not in self.amino_acids or aa2 not in self.amino_acids:
            return 0.0
            
        aa1_idx = self.amino_acids.index(aa1)
        aa2_idx = self.amino_acids.index(aa2)
        blosum_score = self.blosum62[aa1][aa2_idx]
        
        # Normalize BLOSUM62 score to [0, 1] 
        blosum62_min = -4  # Minimum value in BLOSUM62
        blosum62_max = 11  # Maximum value in BLOSUM62
        normalized_score = (blosum_score - blosum62_min) / (blosum62_max - blosum62_min)
        return normalized_score
    
    def _calculate_distance(self, seq1: str, seq2: str) -> float:
        """Calculate distance between two sequences"""
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have the same length")
            
        total_similarity = 0.0
        for i in range(len(seq1)):
            total_similarity += self._calculate_similarity(seq1[i], seq2[i])
            
        # Distance = 1 - (average similarity)
        distance = 1.0 - (total_similarity / len(seq1))
        return distance

    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using KNN"""
        if not self.is_built:
            raise ValueError("KNN not built. Call build() first")
        
        if len(sequence) != self.window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) does not match window size ({self.window_size})")
        
        # Calculate distances to all training sequences
        distances = []
        for train_seq in self.training_sequences:
            distance = self._calculate_distance(sequence, train_seq)
            distances.append(distance)
        
        # Sort distances and get k nearest neighbours
        distances.sort()
        k = min(self.k, len(distances))
        k_nearest_distances = distances[:k]
        
        # KNN score = average similarity of k nearest neighbours
        k_nearest_similarities = [dist for dist in k_nearest_distances]
        return np.mean(k_nearest_similarities)


class SMI(BaseMatrix):
    """
    Substitution Matrix Index-based Scoring Function (SMI)
    Uses 10 different substitution matrices (5 BLOSUM + 5 PAM)
    """
    
    def __init__(self, pseudocount: float = 0.01):
        super().__init__('SMI', pseudocount)
        self.matrix_names = [
            'blosum100', 'blosum75', 'blosum62', 'blosum45', 'blosum30',
            'PAM500', 'PAM400', 'PAM300', 'PAM120', 'PAM30'
        ]
        self.matrices = self._load_all_matrices()
        print(f"SMI Analyzer initialized with {len(self.matrix_names)} matrices")
    
    def _load_all_matrices(self) -> Dict:
        """Load all 10 substitution matrices (5 BLOSUM + 5 PAM)"""
        matrices = {}
        
        # Load BLOSUM matrices
        matrices['blosum100'] = self.blosumDict['blosum100']
        matrices['blosum75'] = self.blosumDict['blosum75']
        matrices['blosum62'] = self.blosumDict['blosum62']
        matrices['blosum45'] = self.blosumDict['blosum45']
        matrices['blosum30'] = self.blosumDict['blosum30']
        
        # Load PAM matrices
        matrices['PAM500'] = self.PAMDict['PAM500']
        matrices['PAM400'] = self.PAMDict['PAM400']
        matrices['PAM300'] = self.PAMDict['PAM300']
        matrices['PAM120'] = self.PAMDict['PAM120']
        matrices['PAM30'] = self.PAMDict['PAM30']
        
        return matrices

    def build(self, windows_df: pd.DataFrame, sequence_col: str = 'sequence', 
              label_col: str = 'known_cleavage_site') -> None:
        """Build SMI training set from cleavage windows"""
        positive_mask = windows_df[label_col] == 1
        positive_sequences = windows_df[positive_mask][sequence_col].tolist()
        
        if not positive_sequences:
            raise ValueError("No positive sample sequences found")
        
        self.window_size = len(positive_sequences[0])
        self.training_sequences = [seq for seq in positive_sequences if len(seq) == self.window_size]
        self.is_built = True
        
        print(f"Built SMI training set: Window size={self.window_size}, Training sequences={len(self.training_sequences)}")
        print(f"Available matrices: {', '.join(self.matrix_names)}")
    
    def _get_matrix_score(self, matrix_name: str, aa1: str, aa2: str) -> float:
        """Get substitution score for amino acid pair from specified matrix"""
        matrix = self.matrices[matrix_name]
        
        if aa1 not in self.amino_acids or aa2 not in self.amino_acids:
            return 0.0
        
        try:
            aa2_idx = self.amino_acids.index(aa2)
            score = matrix[aa1][aa2_idx]
            return float(score)
        except (ValueError, IndexError, KeyError):
            return 0.0
    
    def _calculate_sequence_similarity(self, query_seq: str, train_seq: str, matrix_name: str) -> float:
        """Calculate normalized similarity score between query and training sequence"""
        if len(query_seq) != len(train_seq):
            raise ValueError("Query and training sequences must have the same length")
        
        numerator = 0.0
        denominator = 0.0
        
        for i in range(len(query_seq)):
            query_aa = query_seq[i]
            train_aa = train_seq[i]
            
            cross_score = self._get_matrix_score(matrix_name, query_aa, train_aa)
            numerator += cross_score
            
            self_score = self._get_matrix_score(matrix_name, train_aa, train_aa)
            denominator += self_score
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_smi_score_single_matrix(self, query_seq: str, matrix_name: str) -> float:
        """Calculate SMI score for a single matrix"""
        if not self.training_sequences:
            raise ValueError("No training sequences available. Call build() first.")
        
        max_score = 0.0
        
        for train_seq in self.training_sequences:
            similarity = self._calculate_sequence_similarity(query_seq, train_seq, matrix_name)
            max_score = max(max_score, similarity)
        
        return max_score

    def score_sequence(self, sequence: str, only_average_score: bool = True):
        """Score a sequence using SMI method"""
        if not self.is_built:
            raise ValueError("SMI not built. Call build() first")
        
        if len(sequence) != self.window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) does not match window size ({self.window_size})")
        
        scores = {}
        
        for matrix_name in self.matrix_names:
            smi_score = self._calculate_smi_score_single_matrix(sequence, matrix_name)
            scores[matrix_name] = smi_score
        
        if only_average_score:
            return np.mean(list(scores.values()))
        else:
            return scores
    
    def _generate_smi_score_dataframe(self, windows_df: pd.DataFrame,
                               sequence_col: str = 'sequence',
                               include_original_cols: bool = True) -> pd.DataFrame:
        """Generate a dataframe with scores from all SMI matrices"""
        
        valid_windows = windows_df.copy()
        print(f"Generating SMI scores for {len(valid_windows)} sequences...")
        
        # Generate score columns for each matrix
        for matrix_name in self.matrix_names:
            col_name = f"SMI_{matrix_name.upper()}"
            scores = []
            
            for seq in valid_windows[sequence_col]:
                score = self._calculate_smi_score_single_matrix(seq, matrix_name)
                scores.append(score)
            
            valid_windows[col_name] = scores
            print(f"  ✅ {col_name}: done")
        
        # Add average SMI score
        smi_cols = [f"SMI_{matrix_name.upper()}" for matrix_name in self.matrix_names]
        valid_windows['SMI_AVERAGE'] = valid_windows[smi_cols].mean(axis=1)
        
        print(f"\n🎉 SMI score dataframe generation completed!")
        print(f"   New columns: {smi_cols + ['SMI_AVERAGE']}")
        print(f"   Data shape: {valid_windows.shape}")
        
        if not include_original_cols:
            return valid_windows[[sequence_col] + smi_cols + ['SMI_AVERAGE']]
        
        return valid_windows
   