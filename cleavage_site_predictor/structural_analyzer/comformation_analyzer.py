from sequence_models.pdb_utils import parse_PDB, process_coords
import numpy as np
import pandas as pd
import hashlib
from typing import List, Dict

from .structure_paser import StructureParser
from .structure_downloader import StructureDownloader
from .structure_downloader import StructureFormatter
from .cache_manager import CacheManager


class ConformationAnalyzer(CacheManager):
    """
    ConformationAnalyzer - responsible for extracting conformation features from protein structures
    
    Uses sequence_models.pdb_utils for:
    - CB-CB distances between residues
    - Omega, theta, phi angles
    """
    
    def __init__(self, cache_dir: str = "structure_temp/conformation_cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = True):
        """Initialize Conformation Analysis with a dual caching system."""
        super().__init__(cache_dir=cache_dir, use_disk_cache=use_disk_cache, use_ram_cache=use_ram_cache)
        self.downloader = StructureDownloader()
        self.formatter = StructureFormatter()
        self.parser = StructureParser()

    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key. Overrides the parent method to ensure correctness."""
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_conformation_cache_key(self, uniprot_id: str, pdb_id: str = None, 
                                   structure_source: str = "alphafold") -> str:
        """Generate a unique cache key for conformation analysis."""
        return self._generate_cache_key(
            uniprot_id=uniprot_id,
            pdb_id=pdb_id, 
            structure_source=structure_source,
            analysis_type="conformation"
        )

    def _get_or_create_pdb_cache(self, uniprot_id: str, pdb_id: str = None, 
                                structure_source: str = "alphafold"):
        """Get PDB path from cache or create a new one."""
        pdb_cache_key = self._generate_cache_key(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source, 
            analysis_type="pdb_meta"
        )
        
        cached_data = self._get_cached_data(pdb_cache_key, "pdb")
        if cached_data and 'filtered_pdb_path' in cached_data and 'selected_chain' in cached_data:
            return cached_data['filtered_pdb_path'], cached_data['selected_chain']
            
        print(f" Creating new PDB cache for conformation analysis: {uniprot_id}")
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

    def _parse_pdb(self, pdb_path: str):
        """
        Parse PDB file and return coordinates and residue sequence.

        Args:
            pdb_path: Path to PDB file

        Returns:
            coords: numpy array with shape (n_residues, 3, 3) containing N, CA, C coordinates
            seq: list of residue sequences
        """
        try:
            coords, seq, _ = parse_PDB(pdb_path)
            return coords, seq
        except Exception as e:
            print(f"Error parsing PDB file {pdb_path}: {str(e)}")
            return None, None
    
    def _process_coords(self, coords):
        """
        Process coordinates and return distance, omega, theta, phi.

        Args:
            coords: numpy array with shape (n_residues, 3, 3)

        Returns:
            dist: numpy array, CB-CB distances between consecutive residues  
            omega: numpy array, omega dihedral angles
            theta: numpy array, theta angles
            phi: numpy array, phi dihedral angles
        """
        try:
            # Reshape coordinates for sequence_models format
            coords_dict = {
                'N': coords[:, 0],   # N atoms coordinates
                'CA': coords[:, 1],  # CA atoms coordinates  
                'C': coords[:, 2]    # C atoms coordinates
            }
            
            dist, omega, theta, phi = process_coords(coords_dict)
            return dist, omega, theta, phi
        except Exception as e:
            print(f"Error processing coordinates: {str(e)}")
            return None, None, None, None

    def _calculate_full_sequence_conformation(self, pdb_path: str) -> Dict:
        """Calculate conformation features for the full sequence using actual PDB residue numbers."""
        coords, seq = self._parse_pdb(pdb_path)
        
        if coords is None or seq is None:
            return {}
        
        # Process coordinates to get conformation features
        dist, omega, theta, phi = self._process_coords(coords)
        
        if dist is None:
            return {}
        
        # Get actual residue numbers from filtered PDB file
        residue_numbers = self._extract_residue_numbers_from_pdb(pdb_path)
        
        if not residue_numbers or len(residue_numbers) != len(seq):
            print(f"Warning: Residue number mismatch. Found {len(residue_numbers)} numbers vs {len(seq)} residues")
            return {}
        
        # Convert to per-residue dictionary using actual PDB residue numbers
        conformation_dict = {}
        
        for i, residue_num in enumerate(residue_numbers):
            conformation_dict[residue_num] = {
                'residue': seq[i] if i < len(seq) else 'X',
                'cb_cb_distance': dist[i] if i < len(dist) else np.nan,
                'omega_angle': omega[i] if i < len(omega) else np.nan, 
                'theta_angle': theta[i] if i < len(theta) else np.nan,
                'phi_angle': phi[i] if i < len(phi) else np.nan
            }
        
        return conformation_dict
    
    def _extract_residue_numbers_from_pdb(self, pdb_path: str) -> List[int]:
        """Extract actual residue numbers from filtered PDB file (already contains only one chain)."""
        residue_numbers = []
        seen_residues = set()
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # Extract residue number from PDB line
                        residue_num = int(line[22:26].strip())
                        
                        # No need to check chain - file is already filtered to one chain
                        if residue_num not in seen_residues:
                            seen_residues.add(residue_num)
                            residue_numbers.append(residue_num)
            
            # Sort to ensure consistent ordering
            residue_numbers.sort()
            return residue_numbers
            
        except Exception as e:
            print(f"Error extracting residue numbers from {pdb_path}: {str(e)}")
            return []

    def _get_or_create_conformation_cache(self, uniprot_id: str, pdb_id: str = None, 
                                         structure_source: str = "alphafold") -> Dict:
        """Get or create cache for full sequence conformation data."""
        cache_key = self._get_conformation_cache_key(uniprot_id, pdb_id, structure_source)
        
        cached_data = self._get_cached_data(cache_key, "conformation")
        if cached_data is not None:
            return cached_data
            
        print(f" Calculating new conformation cache: {uniprot_id}")
        filtered_pdb_path, _ = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        
        if not filtered_pdb_path:
            return {}
            
        conformation_properties = self._calculate_full_sequence_conformation(filtered_pdb_path)
        self._cache_data(cache_key, conformation_properties, "conformation")
        
        return conformation_properties # 1-based index

    def run_conformation_analysis(self, uniprot_id: str, 
                                sites: List[int] = None, # 1-based index
                                pdb_id: str = None, 
                                structure_source: str = "alphafold") -> pd.DataFrame:
        """
        Calculate conformation features for specified sites or the full sequence.
        
        Args:
            uniprot_id: UniProt ID
            sites: Optional list of sites to analyze. If None, analyzes the full sequence.
            pdb_id: Optional, specify the PDB ID
            structure_source: Structure source ("pdb" or "alphafold")
            
        Returns:
            DataFrame containing conformation data with columns:
            - position/site: residue position
            - chain: chain identifier  
            - residue: amino acid
            - cb_cb_distance: CB-CB distance to next residue
            - omega_angle: omega dihedral angle
            - theta_angle: theta angle
            - phi_angle: phi dihedral angle
        """
        
        full_conformation_props = self._get_or_create_conformation_cache(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source
        ) # 1-based index
        
        if not full_conformation_props:
            print("Failed to get conformation properties")
            return pd.DataFrame()
        
        _, selected_chain = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        
        positions_to_process = sites if sites is not None else sorted(full_conformation_props.keys()) # 1-based index

        
        results_list = []
        for pos in positions_to_process:
            if pos in full_conformation_props:
                conformation_data = full_conformation_props[pos]
                results_list.append({
                    'position': pos, # 1-based index
                    'chain': selected_chain,
                    'residue': conformation_data.get('residue', 'X'),
                    'cb_cb_distance': conformation_data.get('cb_cb_distance', np.nan),
                    'omega_angle': conformation_data.get('omega_angle', np.nan),
                    'theta_angle': conformation_data.get('theta_angle', np.nan),
                    'phi_angle': conformation_data.get('phi_angle', np.nan)
                })
            else:
                print(f'Site {pos} cannot be found in the sequence')

        if not results_list:
            print(f"No valid conformation data found for the requested positions.")
            return pd.DataFrame()
        
        df = pd.DataFrame(results_list)
        if sites is not None:
            df = df.rename(columns={'position': 'site'})
            
        return df