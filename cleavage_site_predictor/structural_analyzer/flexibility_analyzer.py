import pandas as pd
import numpy as np
from typing import List, Dict
import hashlib

from .structure_paser import StructureParser
from .structure_downloader import StructureDownloader
from .structure_downloader import StructureFormatter
from .cache_manager import CacheManager

class FlexibilityAnalysis(CacheManager):
    def __init__(self, cache_dir: str = "structure_temp/flexibility_cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = True):
        """Initialize Flexibility Analysis with a dual caching system."""
        super().__init__(cache_dir=cache_dir, use_disk_cache=use_disk_cache, use_ram_cache=use_ram_cache)
        self.downloader = StructureDownloader()
        self.formatter = StructureFormatter()
        self.parser = StructureParser()

    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key. Overrides the parent method to ensure correctness.
        """
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_flexibility_cache_key(self, uniprot_id: str, pdb_id: str = None, 
                                   structure_source: str = "alphafold") -> str:
        """Generate a unique cache key for flexibility analysis."""
        return self._generate_cache_key(
            uniprot_id=uniprot_id,
            pdb_id=pdb_id, 
            structure_source=structure_source,
            analysis_type="flexibility"
        )

    def _get_or_create_pdb_cache(self, uniprot_id: str, pdb_id: str = None, 
                                structure_source: str = "alphafold"):
        """Get PDB path from cache or create a new one. This is a helper and mirrors the one in DSSPAnalysis."""
        # This key is for caching the PDB path and chain, not the flexibility data itself
        pdb_cache_key = self._generate_cache_key(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source, 
            analysis_type="pdb_meta"
        )
        
        cached_data = self._get_cached_data(pdb_cache_key, "pdb")
        if cached_data and 'filtered_pdb_path' in cached_data and 'selected_chain' in cached_data:
            return cached_data['filtered_pdb_path'], cached_data['selected_chain']
            
        print(f"🔄 Creating new PDB cache for flexibility analysis: {uniprot_id}")
        valid_chains, valid_pdb = self.downloader.read_pdb_by_source(
            structure_source=structure_source, 
            uniprot_id=uniprot_id, pdb_id=pdb_id, chain=None)
            
        if not valid_chains:
            return None, None
            
        selected_chain = list(valid_chains)[0]
        temp_pdb = self.parser._filter_pdb_by_chain(valid_pdb, selected_chain)    
        filtered_pdb_path = self.parser._save_filtered_pdb(temp_pdb)
        
        cache_data = {'filtered_pdb_path': filtered_pdb_path, 'selected_chain': selected_chain}
        self._cache_data(pdb_cache_key, cache_data, "pdb")
        
        return filtered_pdb_path, selected_chain

    def _calculate_full_sequence_flexibility(self, pdb_path: str) -> Dict[int, float]:
        """Extract the average B-factor or pLDDT for each residue in the PDB file."""
        residue_b_factors = {}
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        try:
                            res_num = int(line[22:26].strip())
                            b_factor = float(line[60:66].strip())
                            if res_num not in residue_b_factors:
                                residue_b_factors[res_num] = []
                            residue_b_factors[res_num].append(b_factor)
                        except (ValueError, IndexError):
                            # Ignore lines with malformed numbers or structure
                            continue
            
            flexibility_dict = {res_num: round(np.mean(b_factors), 2) for res_num, b_factors in residue_b_factors.items()}
            return flexibility_dict

        except Exception as e:
            print(f"Error processing B-factors from {pdb_path}: {str(e)}")
            return {}

    def _get_or_create_flexibility_cache(self, uniprot_id: str, pdb_id: str = None, structure_source: str = "alphafold") -> Dict:
        """Get or create cache for full sequence flexibility data."""
        cache_key = self._get_flexibility_cache_key(uniprot_id, pdb_id, structure_source)
        
        cached_data = self._get_cached_data(cache_key, "flexibility")
        if cached_data is not None:
            return cached_data
            
        print(f"🔄 Calculating new flexibility cache: {uniprot_id}")
        filtered_pdb_path, _ = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        
        if not filtered_pdb_path:
            return {}
            
        flexibility_properties = self._calculate_full_sequence_flexibility(filtered_pdb_path)
        self._cache_data(cache_key, flexibility_properties, "flexibility")
        
        return flexibility_properties


    def run_flexibility_analysis(self, uniprot_id: str, 
                               sites: List[int] = None,
                               pdb_id: str = None, 
                               structure_source: str = "alphafold") -> pd.DataFrame:
        """
        This function calculates flexibility (pLDDT or B-factor) for specified sites or the full sequence.
        
        Args:
            uniprot_id: UniProt ID
            sites: Optional list of sites to analyze. If None, analyzes the full sequence.
            pdb_id: Optional, specify the PDB ID
            structure_source: Structure source ("pdb" or "alphafold")
            
        Returns:
            A DataFrame containing flexibility data.
        """
        
        full_flex_props = self._get_or_create_flexibility_cache(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source
        )
        
        if not full_flex_props:
            print("Failed to get flexibility properties")
            return pd.DataFrame()
        
        _, selected_chain = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        flexibility_type = "pLDDT" if structure_source == "alphafold" else "b_factor"
        
        positions_to_process = sites if sites is not None else sorted(full_flex_props.keys())
        
        results_list = []
        for pos in positions_to_process:
            if pos in full_flex_props:
                results_list.append({
                    'position': pos,
                    'chain': selected_chain,
                    flexibility_type: full_flex_props[pos],
                })
            else:
                print(f'Site {pos} cannot be found in the sequence')

        if not results_list:
            print(f"No valid flexibility data found for the requested positions.")
            return pd.DataFrame()
        
        df = pd.DataFrame(results_list)
        if sites is not None:
            df = df.rename(columns={'position': 'site'})
            
        return df
