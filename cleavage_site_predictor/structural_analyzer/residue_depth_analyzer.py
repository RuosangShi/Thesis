'''
This module contains the ResidueDepthAnalysis class
Use Biopython's ResidueDepth package to calculate the residue depth of proteins
'''

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from typing import List, Dict, Tuple
import os
import hashlib

from .distance_analyzer import DistanceAnalysis
from .cache_manager import CacheManager
from .structure_downloader import StructureDownloader
from .structure_paser import StructureParser
from .structure_downloader import StructureFormatter


class ResidueDepthAnalysis(CacheManager):
    def __init__(self, cache_dir: str = "structure_temp/depth_cache",
                 use_disk_cache: bool = True, use_ram_cache: bool = True, 
                 msms_exec: str = "msms"):
        """Use Dual cache systeme for ResidueDepthAnalysis(RAM + Disk)"""
        super().__init__(cache_dir=cache_dir, use_disk_cache=use_disk_cache, use_ram_cache=use_ram_cache)
        self.distance_analysis = DistanceAnalysis()
        self.downloader = StructureDownloader()
        self.formatter = StructureFormatter()
        self.parser = StructureParser()
        self.msms_exec = msms_exec
        
        print(f"ResidueDepthAnalysis initialized, MSMS executable: {msms_exec}")

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key. Overrides the parent method if necessary."""
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_depth_cache_key(self, uniprot_id: str, pdb_id: str = None, 
                           structure_source: str = "alphafold") -> str:
        """Generate a cache key for residue depth analysis"""
        return self._generate_cache_key(
            uniprot_id=uniprot_id,
            pdb_id=pdb_id, 
            structure_source=structure_source,
            analysis_type="residue_depth"
        )

    def _get_or_create_pdb_cache(self, uniprot_id: str, pdb_id: str = None, 
                                structure_source: str = "alphafold") -> Tuple[str, str]:
        """Get or create PDB cache (using the parent's dual cache system)"""
        pdb_cache_key = self._generate_cache_key(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source, 
            analysis_type="pdb_meta"
        )
        
        # Try to get data from dual cache
        cached_data = self._get_cached_data(pdb_cache_key, "pdb")
        if cached_data and 'filtered_pdb_path' in cached_data and 'selected_chain' in cached_data:
            if os.path.exists(cached_data.get('filtered_pdb_path', '')):
                return cached_data['filtered_pdb_path'], cached_data['selected_chain']
            
        # Create new PDB cache
        print(f" Create new PDB cache for residue depth analysis: {uniprot_id}")
        valid_chains, valid_pdb = self.downloader.read_pdb_by_source(
            structure_source=structure_source, 
            uniprot_id=uniprot_id, pdb_id=pdb_id, chain=None)
            
        if not valid_chains:
            return None, None
            
        # Use the first valid chain
        selected_chain = list(valid_chains)[0]
        temp_pdb = self.parser._filter_pdb_by_chain(valid_pdb, selected_chain)    
        filtered_pdb_path = self.parser._save_filtered_pdb(temp_pdb)
        
        # Prepare cache data
        cache_data = {'filtered_pdb_path': filtered_pdb_path, 'selected_chain': selected_chain}
        
        # Save to dual cache
        self._cache_data(pdb_cache_key, cache_data, "pdb")
        
        return filtered_pdb_path, selected_chain

    '''Get the residue depth results for the full sequence'''
    def _calculate_full_sequence_depth(self, pdb_path: str) -> Dict[int, Dict]:
        """Calculate the residue depth for the full sequence"""
        # Format the PDB file
        adjusted_file_path = f"{pdb_path.rsplit('.', 1)[0]}_adjusted.pdb"
        self.formatter.format_pdb(pdb_path, adjusted_file_path)

        try:
            # Parse the structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', adjusted_file_path)
            model = structure[0]

            # Use Biopython's ResidueDepth to calculate the residue depth
            rd = ResidueDepth(model, msms_exec=self.msms_exec)

            # Extract the residue depth properties to a dictionary
            depth_properties = {}
            absolute_depths = []
            
            # First pass: collect all absolute depths
            for key in rd.keys():
                chain_id, res_id = key
                # Safely extract residue number - res_id could be an int or tuple
                if isinstance(res_id, tuple):
                    residue_num = res_id[1]  # Get the residue number from tuple
                else:
                    residue_num = res_id  # Direct int value
                    
                residue_depth_value, ca_depth_value = rd[key]
                
                depth_properties[residue_num] = {
                    'residue_depth': residue_depth_value,
                    'ca_depth': ca_depth_value
                }
                absolute_depths.append(residue_depth_value)
            
            # Calculate maximum depth for relative depth calculation
            max_depth = max(absolute_depths) if absolute_depths else 1.0
            
            # Second pass: add relative depths
            for residue_num in depth_properties.keys():
                absolute_depth = depth_properties[residue_num]['residue_depth']
                relative_depth = absolute_depth / max_depth if max_depth > 0 else 0.0
                depth_properties[residue_num]['relative_depth'] = relative_depth
                depth_properties[residue_num]['max_depth_protein'] = max_depth

            return depth_properties # 1-based index

        except Exception as e:
            print(f"Warning: Residue depth calculation failed - {str(e)}")
            return {}
        finally:
            # Clean up the adjusted PDB file
            if os.path.exists(adjusted_file_path):
                try:
                    os.remove(adjusted_file_path)
                except OSError:
                    pass

    '''Get or create the full sequence residue depth cache'''
    def _get_or_create_full_depth_cache(self, uniprot_id: str, pdb_id: str = None, 
                                       structure_source: str = "alphafold") -> Dict:
        """Get or create the full sequence residue depth analysis cache (using the parent's dual cache system)"""
        cache_key = self._get_depth_cache_key(uniprot_id, pdb_id, structure_source)
        
        # Try to get data from dual cache
        cached_data = self._get_cached_data(cache_key, "residue_depth")
        if cached_data is not None:
            return cached_data
            
        # Create new residue depth cache
        print(f" Create new residue depth cache for {uniprot_id}")
        filtered_pdb_path, selected_chain = self._get_or_create_pdb_cache(
            uniprot_id, pdb_id, structure_source)
            
        if not filtered_pdb_path:
            return {}
            
        depth_properties = self._calculate_full_sequence_depth(filtered_pdb_path)
        self._cache_data(cache_key, depth_properties, "residue_depth")
        
        return depth_properties

    def run_residue_depth_analysis(self, uniprot_id: str, 
                                  sites: List[int] = None,
                                  pdb_id: str = None, 
                                  structure_source: str = "alphafold") -> pd.DataFrame:
        """
        Calculate the residue depth for the specified sites or the full sequence.
        
        Args:
            uniprot_id: UniProt ID
            sites: Optional list of sites. If None, analyze the full sequence.
            pdb_id: Optional, specify the PDB ID
            structure_source: Structure source ("pdb" or "alphafold")
            
        Returns:
            DataFrame with residue depth data including:
            - residue_depth: absolute residue depth (Å)
            - ca_depth: C-alpha depth (Å)  
            - relative_depth: residue depth / max depth in protein (0-1)
            - max_depth_protein: maximum depth found in this protein (Å)
        """
        
        full_depth_props = self._get_or_create_full_depth_cache(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source
        )
        
        if not full_depth_props:
            print("Failed to get residue depth properties")
            return pd.DataFrame()
        
        # Get chain information (from PDB cache)
        _, selected_chain = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        
        # Determine which positions are included in the output
        positions_to_process = sites if sites is not None else sorted(full_depth_props.keys())
        
        results_list = []
        for pos in positions_to_process:
            if pos in full_depth_props:
                depth_data = full_depth_props[pos]
                results_list.append({
                    'position': pos,
                    'chain': selected_chain,
                    'residue_depth': depth_data['residue_depth'],
                    'ca_depth': depth_data['ca_depth'],
                    'relative_depth': depth_data['relative_depth'],
                    'max_depth_protein': depth_data['max_depth_protein'],
                })
            else:
                print(f'Position {pos} not found in the residue depth analysis sequence')

        if not results_list:
            print(f"No valid residue depth data found for the requested positions")
            return pd.DataFrame()
        
        # If specific sites are requested, rename the 'position' column to 'site'
        df = pd.DataFrame(results_list)
        if sites is not None:
            df = df.rename(columns={'position': 'site'})
            
        return df
    
