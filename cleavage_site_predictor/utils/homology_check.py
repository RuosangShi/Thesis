import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import substitution_matrices, PairwiseAligner
from typing import Dict, Any
from cleavage_site_predictor.window_slicer import WindowSlicer

class HomologyCheck(WindowSlicer):
    def __init__(self,
                 uniprot_timeout: int = 30, 
                 add_cache: bool = True, cache_dir: str = "./uniprot_cache"):
        super().__init__(uniprot_timeout=uniprot_timeout, add_cache=add_cache, cache_dir=cache_dir)

    def check_homology(self,
                        df: pd.DataFrame,
                        uniprot_id_col: str = 'final_entry',
                        heat_map: bool = True):
        '''
        Check the homology of the proteins, return a matrix for the alignment of pairwise protein and optionally a heatmap.
        Global alignment with BLOSUM62.
        
        Args:
            df: DataFrame containing protein information
            uniprot_id_col: Column name containing UniProt IDs
            heat_map: Whether to display a heatmap of the similarity matrix
            
        Returns:
            pd.DataFrame: Similarity matrix
        '''
        # load BLOSUM62 matrix
        blosum62 = substitution_matrices.load("BLOSUM62")
        
        def _calculate_sequence_identity_with_blosum62(seq1: str, seq2: str,
                                                       substitution_matrix) -> Dict[str, Any]:
            """Calculate sequence similarity using BLOSUM62 matrix with global alignment"""
            try:
                # calculate the maximum possible score for seq1
                max_possible_score = 0
                for aa in seq1:
                    try:
                        if aa in substitution_matrix.alphabet:
                            max_possible_score += substitution_matrix[aa, aa]
                    except Exception as e:
                        print(f"Error processing amino acid {aa}: {str(e)}")
                        continue
                
                # set the aligner
                aligner = PairwiseAligner()
                aligner.mode = 'global'
                # Use very small gap penalty to avoid combinatorial explosion
                # while maintaining essentially no gap penalty behavior
                aligner.open_gap_score = -10
                aligner.extend_gap_score = -0.5
                aligner.substitution_matrix = substitution_matrix
                
                # perform alignment
                alignments = aligner.align(seq1, seq2)
                if len(alignments) > 0:
                    alignment = alignments[0]
                    alignment_score = alignment.score
                else:
                    alignment_score = 0
                
                # calculate the score percentage
                score_percentage = (alignment_score / max_possible_score) * 100 if max_possible_score > 0 else 0

                return {
                    "score_percentage": round(score_percentage, 2),
                    "alignment_score": round(alignment_score, 2),
                    "max_possible_score": round(max_possible_score, 2)
                }
            except Exception as e:
                print(f"BLOSUM62 scoring error: {str(e)}")
                return {"error": f"Scoring failed: {str(e)}"}
        
        # Get unique UniProt IDs
        uniprot_ids = df[uniprot_id_col].unique().tolist()
        n_proteins = len(uniprot_ids)
        
        print(f"Calculating homology for {n_proteins} proteins...")
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_proteins, n_proteins))
        
        # Store sequences to avoid repeated downloads
        sequences = {}
        
        # Get all sequences first
        for uniprot_id in uniprot_ids:
            try:
                sequences[uniprot_id] = self._get_sequence_for_uniprot(uniprot_id)
            except Exception as e:
                print(f"Failed to get sequence for {uniprot_id}: {str(e)}")
                sequences[uniprot_id] = ""  # Empty sequence as fallback
        
        # Calculate pairwise similarities
        for i, uniprot_id1 in enumerate(uniprot_ids):
            for j, uniprot_id2 in enumerate(uniprot_ids):
                if i == j:
                    # Same protein - 100% similarity
                    similarity_matrix[i, j] = 100.0
                elif i < j:
                    # Calculate similarity for upper triangle only
                    seq1 = sequences[uniprot_id1]
                    seq2 = sequences[uniprot_id2]
                    
                    if seq1 and seq2:  # Both sequences are available
                        result = _calculate_sequence_identity_with_blosum62(seq1, seq2, blosum62)
                        if "error" not in result:
                            similarity_score = result["score_percentage"]
                            if similarity_score > 40:
                                print(f"  {uniprot_id1} vs {uniprot_id2}: {similarity_score:.2f}%")
                        else:
                            similarity_score = 0.0
                            print(f"  {uniprot_id1} vs {uniprot_id2}: Error - {result['error']}")
                    else:
                        similarity_score = 0.0
                        print(f"  {uniprot_id1} vs {uniprot_id2}: Missing sequence data")
                    
                    similarity_matrix[i, j] = similarity_score
                    # Mirror to lower triangle
                    similarity_matrix[j, i] = similarity_score
        
        # Create DataFrame for the similarity matrix
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=uniprot_ids, 
            columns=uniprot_ids
        )
        
        print(f"\nHomology analysis completed!")
        
        # Create heatmap if requested
        if heat_map:
            plt.figure(figsize=(12, 10))
                
            # Create heatmap
            sns.heatmap(
                    similarity_df, 
                    annot=True, 
                    fmt='.1f', 
                    cmap='viridis',
                    square=True,
                    cbar_kws={'label': 'Sequence Similarity (%)'},
                    vmin=0,
                    vmax=100
            )
                
            plt.title(f'Protein Sequence Homology Matrix\n(BLOSUM62, Global Alignment)', 
                         fontsize=14, fontweight='bold')
            plt.xlabel('UniProt ID', fontsize=12)
            plt.ylabel('UniProt ID', fontsize=12)
                
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
                
            plt.tight_layout()
            plt.show()
        
        return similarity_df

    def generate_blast_format(self,
                            df: pd.DataFrame,
                            uniprot_id_col: str = 'final_entry',
                            output_file: str = None) -> str:
        '''
        Generate FASTA format text for NCBI BLAST analysis.
        
        Args:
            df: DataFrame containing protein information
            uniprot_id_col: Column name containing UniProt IDs
            output_file: Optional file path to save the BLAST format text
            
        Returns:
            str: FASTA formatted text for BLAST
        '''
        # Get unique UniProt IDs
        uniprot_ids = df[uniprot_id_col].unique().tolist()
        n_proteins = len(uniprot_ids)
        
        print(f"Generating BLAST format for {n_proteins} proteins...")
        
        # Store sequences
        sequences = {}
        
        # Get all sequences first
        for uniprot_id in uniprot_ids:
            try:
                sequences[uniprot_id] = self._get_sequence_for_uniprot(uniprot_id)
            except Exception as e:
                print(f"Failed to get sequence for {uniprot_id}: {str(e)}")
                sequences[uniprot_id] = ""  # Empty sequence as fallback
        
        # Generate BLAST format text
        blast_lines = []
        blast_lines.append("# FASTA sequences for NCBI BLAST analysis")
        blast_lines.append("# Paste this text into BLASTP (https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins)")
        blast_lines.append("# This will provide % identity, positives, and e-value statistics")
        blast_lines.append("")
                
        for uniprot_id in uniprot_ids:
            if uniprot_id in sequences and sequences[uniprot_id]:
                # FASTA header with UniProt ID
                blast_lines.append(f">{uniprot_id}")
                # Sequence with 80 characters per line (standard FASTA format)
                sequence = sequences[uniprot_id]
                for i in range(0, len(sequence), 80):
                    blast_lines.append(sequence[i:i+80])
                blast_lines.append("")  # Empty line between sequences
                
        blast_text = "\n".join(blast_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(blast_text)
                print(f"BLAST format text saved to: {output_file}")
            except Exception as e:
                print(f"Failed to save BLAST format text to file: {str(e)}")
        
        # Display summary
        valid_sequences = len([uid for uid in uniprot_ids if uid in sequences and sequences[uid]])
        print(f"\nBLAST format text generated for {valid_sequences} sequences!")
        print("Copy the text below and paste it into NCBI BLASTP:")
        print("="*80)
        print(blast_text)
        print("="*80)
        
        return blast_text

