from .uniprot_extractor import UniprotExtractor
import re
import numpy as np
from Bio import pairwise2

class CleavageSiteFinder(UniprotExtractor):
    """Find cleavage sites in protein sequences"""

    def __init__(self, uniprot_id: str):
        super().__init__(uniprot_id)
        self.sequence = self.get_gene_sequence()


    def get_cleavage_window(self, P1_site: list, window_size: int=14):
        """Get the cleavage window for a given 1-based P1 site"""
        window = []
        # P1 site is 1-based
        for site in P1_site:
            start = max(0, site-window_size//2)
            end = min(len(self.sequence), site+window_size//2)

            left_context = self.sequence[start:site]
            right_context = self.sequence[site:end]
            window.append(str(left_context) + '/' + str(right_context))
        return window


    def find_cleavage_sites(self, logo: str):
        """
        Find the cleavage site for a given logo like "HHQK/LVFF,HH/QKLVFF", 
        retrun 1-based index
        !!!Not alignment, must 100% match!!!
        """
        P1_site = []
        for motif in logo.split(';'):
            motif = motif.strip()
            left, right = motif.split('/', 1)
            try:
                pattern = re.compile(re.escape(left) + re.escape(right))
                # Find all the matched positions
                match = pattern.finditer(self.sequence)
                for m in match:
                    start = m.start()
                    # P1 (1-based)
                    P1_site.append(start + len(left))
            except Exception as e:
                print(f"Error finding cleavage site for {self.uniprot_id} {motif}: {e}")
                P1_site.append(np.nan)
                continue
        return P1_site
    
    def align_cleavage_site(self, logo:str, target_uniprot_id:str, threshold:float=70):
        """
        Fiven a cleavage logo, return a aligned cleavage site postion on the target protein(1-based index)
        Local alignment with identity threshold > 70%
        Disigned for finding human protein cleavage site if only know the mouse cleavage logo
        
        Args:
            logo: cleavage site logo (e.g. "HHQK/LVFF")
            target_uniprot_id: target protein UniProt ID
            threshold: identity threshold percentage (default: 70)
            
        Returns:
            1-based cleavage site position on the target protein
        """
        # Get the target protein sequence
        target_extractor = UniprotExtractor(target_uniprot_id)
        target_sequence = target_extractor.get_gene_sequence()
        
        if not target_sequence:
            return {"error": "Could not get target sequence"}
        
        # Handle multiple cleavage logos
        results = []
        for motif in logo.split(';'):
            try:
                motif = motif.strip()
                if not motif or '/' not in motif:
                    continue
                left, right = motif.split('/', 1)
                full_motif = left + right
                cleavage_pos = len(left)  # 1-based position of cleavage in the motif
                alignment = pairwise2.align.localms(
                    full_motif, target_sequence,
                    2,  # match score
                    -1,  # mismatch penalty
                    -0.5,  # gap open penalty
                    -0.1,  # gap extension penalty
                    one_alignment_only=True
                )
                alignment = alignment[0]
                matches = 0
                aligned_length = 0
                for a, b in zip(alignment.seqA, alignment.seqB):
                    if a != '-' and b != '-':
                        aligned_length += 1
                        if a == b:
                            matches += 1
                identity = (matches / aligned_length) * 100 if aligned_length > 0 else 0
                print(f"alignment in {target_uniprot_id} identity: {identity}")
                if identity < threshold:
                    results.append(np.nan)
                    continue
                
                # calculate the gap count in the query sequence
                gap_count_in_target = alignment.seqB[:cleavage_pos].count('-')
                # calculate the target sequence position (1-based)
                # best_alignment.start is 0-based, cleavage_pos is 1-based
                target_cleavage_pos = alignment.start + cleavage_pos - gap_count_in_target
                results.append(target_cleavage_pos)
            except Exception as e:
                print(f"Error processing motif {motif}: {str(e)}")
                results.append(np.nan)
        return results
        
        
        
    
