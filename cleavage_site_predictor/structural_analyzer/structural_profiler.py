"""
Structural profiler - responsible for extracting distance features from protein windows
"""

import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from .cache_manager import CacheManager
from .distance_analyzer import DistanceAnalysis
from .residue_depth_analyzer import ResidueDepthAnalysis
from .dssp_analyzer import DSSPAnalysis
from .flexibility_analyzer import FlexibilityAnalysis
from .comformation_analyzer import ConformationAnalyzer



class StructuralProfiler(CacheManager):
    """
    Structural profiler for extracting distance features from protein windows
    
    Focuses on distance analysis:
    - Sequence distance to membrane
    - Spatial coordination distance to membrane
    """
    
    def __init__(self, uniprot_timeout: int = 30):
        """
        Initialize the structural profiler
        
        Args:
            uniprot_timeout: Timeout for UniProt API requests (seconds)
        """
        self.uniprot_timeout = uniprot_timeout
        self.distance_analyzer = DistanceAnalysis()
        self.residue_depth_analyzer = ResidueDepthAnalysis()
        self.dssp_analyzer = DSSPAnalysis()
        self.flexibility_analyzer = FlexibilityAnalysis()
        self.conformation_analyzer = ConformationAnalyzer()

    
    def _extract_window_positions(self, center_position: int, window_size: int) -> List[int]:
        """
        Extract positions for each residue in the window
        
        Args:
            center_position: Center position of the window (1-based)
            window_size: Size of the window
            
        Returns:
            List of positions in the window (only valid positions within sequence), 1-based.
            e.g. center_position = 10, window_size = 20, return [1, 2, 3, ..., 20]
        """
        half_window = window_size // 2
        start_pos = center_position - half_window + (1 if window_size % 2 == 0 else 0) # 1-based
        end_pos = center_position + half_window # 1-based
        
        # Only include positions within the sequence bounds
        positions = []
        for pos in range(start_pos, end_pos + 1):
            positions.append(pos)
        
        return positions
    


class DistanceProfiler(StructuralProfiler):
    def __init__(self):
        super().__init__()
        
     
    def profile_distance(self, windows_df: pd.DataFrame,
                               uniprot_id_col: str = 'entry',
                               extracellular_regions_col: str = 'extracellular',
                               transmembrane_regions_col: str = 'transmembrane',
                               intracellular_regions_col: str = 'intracellular',
                               sequence_length_col: str = 'sequence_length',
                               include_sequence_distance: bool = True,
                               include_spatial_distance: bool = True,
                               structure_source: str = "alphafold",
                               show_original_data: bool = True,
                               only_extracellular_sites: bool = True,
                               include_sequence_distance_mean: bool = True,
                               include_spatial_distance_mean: bool = True) -> pd.DataFrame:
        """
        Profile distance features for all windows as vectors
        
        For each window, generates distance vectors where each element corresponds to
        the distance of that position in the window.
        
        Args:
            windows_df: DataFrame containing window information with topology columns
            include_sequence_distance: Whether to include sequence distance analysis
            include_spatial_distance: Whether to include spatial distance analysis  
            structure_source: Structure source for spatial analysis
            show_original_data: If keep the columns from the windows_df
            only_extracellular_sites: If only extracellular sites are considered
        Returns:
            DataFrame with distance vectors added:
            - sequence_distance_vector: [d1, d2, ..., dl] where l = window_size
            - spatial_distance_vector: [d1, d2, ..., dl] where l = window_size
        """
        # Initialize distance vector columns
        if include_sequence_distance:
            windows_df['sequence_distance_vector'] = None
        
        if include_spatial_distance:
            windows_df['spatial_distance_vector'] = None
        
        if include_sequence_distance_mean:
            windows_df['sequence_distance_mean'] = None
        
        if include_spatial_distance_mean:
            windows_df['spatial_distance_mean'] = None
        
        # Process each unique protein
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                # Get topology from first row (should be same for all windows of same protein)
                first_row = entry_windows.iloc[0]
                extracellular_regions = first_row.get(extracellular_regions_col, [])
                transmembrane_regions = first_row.get(transmembrane_regions_col, [])
                intracellular_regions = first_row.get(intracellular_regions_col, [])
                sequence_length = first_row.get(sequence_length_col, None)
                
                if not transmembrane_regions:
                    print(f"Warning: Missing transmembrane information for {entry}")
                    continue
                
                # Handle empty extracellular regions using window_slicer strategy
                if not extracellular_regions:
                    if sequence_length is None:
                        print(f"Warning: Cannot infer extracellular regions without sequence length for {entry}")
                        continue
                    
                    print(f"  No extracellular regions provided for {entry}, inferring...")
                    
                    # Use distance_analyzer's method to get extracellular regions
                    extracellular_regions = self.distance_analyzer.get_extracellular_regions(
                        sequence_length=sequence_length,
                        intracellular_regions=intracellular_regions,
                        transmembrane_regions=transmembrane_regions
                    )
                    
                    print(f"  Inferred {len(extracellular_regions)} extracellular regions")
                
                if not extracellular_regions:
                    print(f"Warning: No extracellular regions found for {entry}")
                    continue
                
                # Process each window for this protein
                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position']
                    window_size = row['window_size']
                    
                    # Extract window positions
                    positions = self._extract_window_positions(center_position, window_size)
                    
                    if not positions:
                        print(f"Warning: No valid positions found for window at {center_position}")
                        continue
                    
                    # Initialize distance vectors with NaN values
                    seq_vector = [np.nan] * window_size
                    spatial_vector = [np.nan] * window_size
                    
                    # Sequence distance analysis
                    if include_sequence_distance:
                        try:
                            seq_results = self.distance_analyzer.distance_analysis(
                                sites=positions, # 1-based index
                                distance_type="sequence",
                                extracellular_regions=extracellular_regions,
                                transmembrane_regions=transmembrane_regions,
                                only_extracellular_sites=only_extracellular_sites
                            )
                            
                            if not seq_results.empty and 'distance' in seq_results.columns and 'site' in seq_results.columns:
                                # Map results to vector positions
                                for _, result_row in seq_results.iterrows():
                                    site = result_row['site']
                                    distance = result_row['distance']
                                    
                                    # Find position in window vector
                                    if site in positions:
                                        vector_idx = positions.index(site)
                                        seq_vector[vector_idx] = distance
                                
                                windows_df.at[idx, 'sequence_distance_vector'] = seq_vector
                                
                            else:
                                print(f"    Warning: No valid sequence distance results")
                            
                        except Exception as e:
                            print(f"    Warning: Sequence distance analysis failed: {str(e)}")
                    
                    # Spatial distance analysis
                    if include_spatial_distance:
                        try:
                            spatial_results = self.distance_analyzer.distance_analysis(
                                sites=positions,
                                uniprot_id=entry,
                                distance_type="coordination",
                                extracellular_regions=extracellular_regions,
                                transmembrane_regions=transmembrane_regions,
                                structure_source=structure_source,
                                only_extracellular_sites=only_extracellular_sites
                            )
                            
                            if not spatial_results.empty and 'distance' in spatial_results.columns and 'site' in spatial_results.columns:
                                # Map results to vector positions
                                for _, result_row in spatial_results.iterrows():
                                    site = result_row['site']
                                    distance = result_row['distance']
                                    
                                    # Find position in window vector
                                    if site in positions:
                                        vector_idx = positions.index(site)
                                        spatial_vector[vector_idx] = distance
                                
                                windows_df.at[idx, 'spatial_distance_vector'] = spatial_vector
                                
                            else:
                                print(f"    Warning: No valid spatial distance results")
                            
                        except Exception as e:
                            print(f"    Warning: Spatial distance analysis failed: {str(e)}")
                
            except Exception as e:
                print(f"Error processing protein {entry}: {str(e)}")
                continue
        
        print(f"\nDistance vector profiling completed")
        
        # Report added features
        added_features = []
        if include_sequence_distance:
            added_features.append('sequence_distance_vector')
        if include_spatial_distance:
            added_features.append('spatial_distance_vector')
        
        print(f"Distance vector features added: {added_features}")
        
        # Handle show_original_data parameter
        if not show_original_data:
            # Keep only the distance vector columns
            columns_to_keep = []
            if include_sequence_distance:
                columns_to_keep.append('sequence_distance_vector')
            if include_spatial_distance:
                columns_to_keep.append('spatial_distance_vector')
            
            if columns_to_keep:
                print(f"Keeping only distance vector columns: {columns_to_keep}")
                windows_df = windows_df[columns_to_keep].copy()
            else:
                print("Warning: No distance vector columns to keep")
        else:
            print(f"Keeping all original columns plus distance vectors")
        
        if include_sequence_distance_mean:
            windows_df['sequence_distance_mean'] = windows_df['sequence_distance_vector'].apply(np.nanmean)
        
        if include_spatial_distance_mean:
            windows_df['spatial_distance_mean'] = windows_df['spatial_distance_vector'].apply(np.nanmean)
        
        return windows_df
    


class ResidueDepthProfiler(StructuralProfiler):
    def __init__(self):
        super().__init__()
        
        
    def profile_residue_depth(self, windows_df: pd.DataFrame,
                             uniprot_id_col: str = 'entry',
                             structure_source: str = "alphafold",
                             show_original_data: bool = True,
                             include_depth_mean: bool = True,
                             include_relative_depth: bool = True,
                             include_ca_depth: bool = False) -> pd.DataFrame:
        """
        Profile residue depth features for all windows as vectors
        
        For each window, generates residue depth vectors where each element corresponds to
        the residue depth of that position in the window.
        
        Args:
            windows_df: DataFrame containing window information
            uniprot_id_col: Column name for UniProt IDs
            structure_source: Structure source for analysis
            show_original_data: Whether to keep the original columns
            include_depth_mean: Whether to include mean residue depth values
            include_relative_depth: Whether to include relative depth vectors and means
            include_ca_depth: Whether to include CA depth vectors and means
            
        Returns:
            DataFrame with residue depth vectors added:
            - residue_depth_vector: [d1, d2, ..., dl] absolute depth (Å)
            - relative_depth_vector: [r1, r2, ..., rl] relative depth (0-1)
            - ca_depth_vector: [c1, c2, ..., cl] CA depth (Å) [optional]
            - residue_depth_mean: Mean absolute residue depth for the window
            - relative_depth_mean: Mean relative depth for the window [optional]
            - ca_depth_mean: Mean CA depth for the window [optional]
        """
        # Initialize residue depth vector columns
        windows_df['residue_depth_vector'] = None
        
        if include_relative_depth:
            windows_df['relative_depth_vector'] = None
            
        if include_ca_depth:
            windows_df['ca_depth_vector'] = None
        
        if include_depth_mean:
            windows_df['residue_depth_mean'] = None
            
        if include_relative_depth and include_depth_mean:
            windows_df['relative_depth_mean'] = None
            
        if include_ca_depth and include_depth_mean:
            windows_df['ca_depth_mean'] = None
        
        # Process each unique protein
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein for residue depth analysis: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                # Calculate full sequence residue depth for efficiency
                full_depth_dict = {}
                
                # Get all residue depth features at once
                depth_results = self.residue_depth_analyzer.run_residue_depth_analysis(
                    uniprot_id=entry,
                    structure_source=structure_source
                )
                if not depth_results.empty and len(depth_results) > 0:
                    try:
                        # Create dictionaries for different depth metrics - check required columns exist
                        required_cols = ['position', 'residue_depth', 'relative_depth', 'ca_depth']
                        if all(col in depth_results.columns for col in required_cols):
                            full_depth_dict = dict(zip(depth_results['position'], depth_results['residue_depth']))
                            full_relative_depth_dict = dict(zip(depth_results['position'], depth_results['relative_depth']))
                            full_ca_depth_dict = dict(zip(depth_results['position'], depth_results['ca_depth']))
                            print(f"    Extracted residue depth data for {len(full_depth_dict)} positions.")
                        else:
                            missing_cols = [col for col in required_cols if col not in depth_results.columns]
                            print(f"    Error: Missing required columns: {missing_cols}")
                            full_depth_dict = {}
                            full_relative_depth_dict = {}
                            full_ca_depth_dict = {}
                        
                        # Safely get max depth information
                        try:
                            if 'max_depth_protein' in depth_results.columns and len(depth_results) > 0:
                                max_depth_val = depth_results['max_depth_protein'].iloc[0]
                                print(f"    Max depth in protein: {max_depth_val:.3f} Å")
                            else:
                                print("    Max depth information not available")
                        except (IndexError, KeyError) as e:
                            print(f"    Warning: Could not extract max depth information: {str(e)}")
                            
                    except Exception as depth_error:
                        print(f"    Error extracting depth data: {str(depth_error)}")
                        full_depth_dict = {}
                        full_relative_depth_dict = {}
                        full_ca_depth_dict = {}
                else:
                    print(f"    Residue depth analysis failed or returned no data for {entry}")
                    full_depth_dict = {}
                    full_relative_depth_dict = {}
                    full_ca_depth_dict = {}
                
                # Process each window for this protein
                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position']
                    window_size = row['window_size']
                    
                    # Extract window positions
                    positions = self._extract_window_positions(center_position, window_size)
                    
                    if not positions:
                        print(f"Warning: No valid positions found for window at {center_position}")
                        continue
                    
                    # Initialize vectors with NaN values
                    depth_vector = [np.nan] * window_size
                    relative_depth_vector = [np.nan] * window_size
                    ca_depth_vector = [np.nan] * window_size
                    
                    # Fill residue depth vectors
                    if full_depth_dict:
                        for i, pos in enumerate(positions):
                            try:
                                if pos in full_depth_dict:
                                    depth_vector[i] = full_depth_dict[pos]
                                if pos in full_relative_depth_dict and include_relative_depth:
                                    relative_depth_vector[i] = full_relative_depth_dict[pos]
                                if pos in full_ca_depth_dict and include_ca_depth:
                                    ca_depth_vector[i] = full_ca_depth_dict[pos]
                            except Exception as pos_error:
                                print(f"    Warning: Error processing position {pos}: {str(pos_error)}")
                                continue
                        
                        windows_df.at[idx, 'residue_depth_vector'] = depth_vector
                        
                        if include_relative_depth:
                            windows_df.at[idx, 'relative_depth_vector'] = relative_depth_vector
                            
                        if include_ca_depth:
                            windows_df.at[idx, 'ca_depth_vector'] = ca_depth_vector
                
            except Exception as e:
                print(f"Error processing protein {entry} for residue depth analysis: {str(e)}")
                continue
        
        print(f"\nResidue depth analysis completed")
        
        # Report added features
        added_features = ['residue_depth_vector']
        if include_relative_depth:
            added_features.append('relative_depth_vector')
        if include_ca_depth:
            added_features.append('ca_depth_vector')
        
        print(f"Added residue depth features: {added_features}")
        
        # Handle show_original_data parameter
        if not show_original_data:
            # Keep only residue depth vector columns
            columns_to_keep = ['residue_depth_vector']
            if include_relative_depth:
                columns_to_keep.append('relative_depth_vector')
            if include_ca_depth:
                columns_to_keep.append('ca_depth_vector')
            if include_depth_mean:
                columns_to_keep.append('residue_depth_mean')
                if include_relative_depth:
                    columns_to_keep.append('relative_depth_mean')
                if include_ca_depth:
                    columns_to_keep.append('ca_depth_mean')
            
            if columns_to_keep:
                print(f"Keeping only residue depth vector columns: {columns_to_keep}")
                windows_df = windows_df[columns_to_keep].copy()
            else:
                print("Warning: No residue depth vector columns to keep")
        else:
            print(f"Keeping all original columns plus residue depth vectors")
        
        # Calculate mean values if requested
        if include_depth_mean and 'residue_depth_vector' in windows_df.columns:
            windows_df['residue_depth_mean'] = windows_df['residue_depth_vector'].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) and any(not np.isnan(v) for v in x if v is not None) else np.nan
            )
            
        if include_relative_depth and include_depth_mean and 'relative_depth_vector' in windows_df.columns:
            windows_df['relative_depth_mean'] = windows_df['relative_depth_vector'].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) and any(not np.isnan(v) for v in x if v is not None) else np.nan
            )
            
        if include_ca_depth and include_depth_mean and 'ca_depth_vector' in windows_df.columns:
            windows_df['ca_depth_mean'] = windows_df['ca_depth_vector'].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) and any(not np.isnan(v) for v in x if v is not None) else np.nan
            )
        
        return windows_df
  

class DSSPProfiler(StructuralProfiler):
    def __init__(self):
        super().__init__()
        
    def profile_dssp(self, windows_df: pd.DataFrame,
                    uniprot_id_col: str = 'entry',
                    include_rsa: bool = True,
                    include_secondary_structure: bool = True,
                    rsa_cal: str = 'Wilke',
                    structure_source: str = "alphafold",
                    show_original_data: bool = True,
                    include_rsa_mean: bool = True) -> pd.DataFrame:
        """
        Profile DSSP features for all windows as vectors
        
        For each window, generates DSSP vectors where each element corresponds to
        the RSA/secondary structure of that position in the window.
        
        Args:
            windows_df: DataFrame containing window information
            uniprot_id_col: Column name for UniProt IDs
            include_rsa: Whether to include RSA (Relative Solvent Accessibility) analysis
            include_secondary_structure: Whether to include secondary structure analysis
            rsa_cal: RSA calculation method ('Wilke', 'Sander', 'Miller')
            structure_source: Structure source for analysis
            show_original_data: Whether to keep the original columns
            include_rsa_mean: Whether to include mean RSA values
            
        Returns:
            DataFrame with DSSP vectors added:
            - rsa_vector: [rsa1, rsa2, ..., rsal] where l = window_size
            - secondary_structure_vector: [ss1, ss2, ..., ssl] where l = window_size
            - rsa_mean: mean RSA value for the window
        """
        # Initialize DSSP vector columns
        if include_rsa:
            windows_df['rsa_vector'] = None
            
        if include_secondary_structure:
            windows_df['secondary_structure_vector'] = None
            windows_df['secondary_structure_encoded'] = None  # Numerical encoding
            
        if include_rsa_mean:
            windows_df['rsa_mean'] = None
        
        # Process each unique protein
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                # Calculate full sequence RSA and secondary structure once for efficiency
                full_rsa_dict = {}
                full_ss_dict = {}
                
                # Get all DSSP features at once
                dssp_results = self.dssp_analyzer.run_dssp_analysis(
                            uniprot_id=entry,
                            rsa_cal=rsa_cal,
                            structure_source=structure_source
                        )
                        
                if not dssp_results.empty:
                    print(f"  DSSP analysis successful for {entry}, {len(dssp_results)} positions found.")
                    if include_rsa:
                        full_rsa_dict = dict(zip(dssp_results['position'], dssp_results['rsa']))
                        print(f"    RSA data extracted for {len(full_rsa_dict)} positions.")
                
                if include_secondary_structure:
                        full_ss_dict = dict(zip(dssp_results['position'], dssp_results['dssp']))
                        print(f"    Secondary structure data extracted for {len(full_ss_dict)} positions.")
                else:
                    print(f"    Warning: DSSP analysis failed or returned no data for {entry}")
                
                # Process each window for this protein
                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position']
                    window_size = row['window_size']
                    
                    # Extract window positions
                    positions = self._extract_window_positions(center_position, window_size)
                    
                    if not positions:
                        print(f"Warning: No valid positions found for window at {center_position}")
                        continue
                    
                    # Initialize vectors with NaN/None values
                    rsa_vector = [np.nan] * window_size
                    ss_vector = [None] * window_size
                    ss_encoded_vector = [np.nan] * window_size
                    
                    # Fill RSA vector
                    if include_rsa and full_rsa_dict:
                        for i, pos in enumerate(positions):
                            if pos in full_rsa_dict:
                                rsa_vector[i] = full_rsa_dict[pos]
                        
                        windows_df.at[idx, 'rsa_vector'] = rsa_vector
                    
                    # Fill secondary structure vector
                    if include_secondary_structure and full_ss_dict:
                        # Secondary structure encoding
                        ss_encoding = {
                            '-': 0, 'B': 1, 'E': 2, 'G': 3, 'H': 4, 
                            'I': 5, 'P': 6, 'S': 7, 'T': 8
                        }
                        
                        for i, pos in enumerate(positions):
                            if pos in full_ss_dict:
                                ss_symbol = full_ss_dict[pos]
                                ss_vector[i] = ss_symbol
                                ss_encoded_vector[i] = ss_encoding.get(ss_symbol, -1)
                        
                        windows_df.at[idx, 'secondary_structure_vector'] = ss_vector
                        windows_df.at[idx, 'secondary_structure_encoded'] = ss_encoded_vector
                
            except Exception as e:
                print(f"Error processing protein {entry}: {str(e)}")
                continue
        
        print(f"\nDSSP profiling completed")
        
        # Report added features
        added_features = []
        if include_rsa:
            added_features.append('rsa_vector')
        if include_secondary_structure:
            added_features.extend(['secondary_structure_vector', 'secondary_structure_encoded'])
        
        print(f"DSSP features added: {added_features}")
        
        # Handle show_original_data parameter
        if not show_original_data:
            # Keep only the DSSP vector columns
            columns_to_keep = []
            if include_rsa:
                columns_to_keep.append('rsa_vector')
            if include_secondary_structure:
                columns_to_keep.extend(['secondary_structure_vector', 'secondary_structure_encoded'])
            
            if columns_to_keep:
                print(f"Keeping only DSSP vector columns: {columns_to_keep}")
                windows_df = windows_df[columns_to_keep].copy()
            else:
                print("Warning: No DSSP vector columns to keep")
        else:
            print(f"Keeping all original columns plus DSSP vectors")
        
        # Calculate mean RSA if requested
        if include_rsa_mean and 'rsa_vector' in windows_df.columns:
            windows_df['rsa_mean'] = windows_df['rsa_vector'].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) and any(not np.isnan(v) for v in x if v is not None) else np.nan
            )
        
        return windows_df
   
    def profile_hbonds(self, windows_df: pd.DataFrame, 
                      uniprot_id_col: str = 'entry', 
                      structure_source: str = "alphafold",
                      threshold: float = None,
                      show_original_data: bool = True,
                      include_hbond_mean: bool = True) -> pd.DataFrame:
        '''
        Profile the hydrogen bonds from DSSP results in the protein structure.
        
        For each window, generates hydrogen bond matrices where each element corresponds to
        the hydrogen bond interactions between residues in the window.
        
        Args:
            windows_df: DataFrame containing window information
            uniprot_id_col: Column name for UniProt IDs
            structure_source: Structure source for analysis
            threshold: Threshold for hydrogen bond energy, if None, all hydrogen bonds are included
            show_original_data: Whether to keep the original columns
            include_hbond_mean: Whether to include mean hydrogen bond values
            
        Returns:
            DataFrame with hydrogen bond matrices added:
            - hbond_energy_matrix: Matrix of hydrogen bond energies
            - hbond_count_matrix: Matrix of hydrogen bond counts
            - nh_o_energy_matrix: Matrix of N-H → O hydrogen bond energies
            - o_nh_energy_matrix: Matrix of O → N-H hydrogen bond energies
            - hbond_energy_mean: Mean hydrogen bond energy for the window
            - hbond_count_mean: Mean hydrogen bond count for the window
        '''
        # Initialize hydrogen bond matrix columns
        windows_df['hbond_energy_matrix'] = None
        windows_df['hbond_count_matrix'] = None
        windows_df['nh_o_energy_matrix'] = None
        windows_df['o_nh_energy_matrix'] = None
        
        if include_hbond_mean:
            windows_df['hbond_energy_mean'] = None
            windows_df['hbond_count_mean'] = None
            windows_df['nh_o_energy_mean'] = None
            windows_df['o_nh_energy_mean'] = None
        
        # Process each unique protein
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein for hydrogen bond analysis: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                # Get full sequence DSSP properties
                dssp_properties = self.dssp_analyzer._get_or_create_full_dssp_cache(
                    uniprot_id=entry,
                    structure_source=structure_source
                )
                
                if not dssp_properties:
                    print(f"    Warning: DSSP properties not found for {entry}")
                    continue
                
                # Generate hydrogen bond matrices for the entire protein
                hbond_matrices = self.dssp_analyzer.generate_hbond_matrices(dssp_properties, threshold=threshold)
                
                if not hbond_matrices:
                    print(f"    Warning: Hydrogen bond matrices not generated for {entry}")
                    continue
                
                # Get position mapping
                all_positions = list(range(1, hbond_matrices['hbond_energy'].shape[0] + 1)) # 1-based index
                pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)} # [1-based index]: 0-based index
                print(f"    Hydrogen bond matrices generated for {len(all_positions)} positions")
                
                # Process each window for this protein
                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position']
                    window_size = row['window_size']
                    
                    # Extract window positions
                    window_positions = self._extract_window_positions(center_position, window_size)
                    
                    if not window_positions:
                        print(f"Warning: No valid positions found for window at {center_position}")
                        continue
                    
                    # Get indices for window positions that exist in the protein
                    valid_indices = []
                    for pos in window_positions:
                        if pos in pos_to_idx:
                            valid_indices.append(pos_to_idx[pos])
                    
                    if not valid_indices:
                        print(f"Warning: No valid indices found for window at {center_position}")
                        continue
                    
                    # Extract submatrices for the window
                    try:
                        # Create window-sized matrices
                        window_hbond_energy = np.full((window_size, window_size), np.nan)
                        window_hbond_count = np.full((window_size, window_size), np.nan)
                        window_nh_o_energy = np.full((window_size, window_size), np.nan)
                        window_o_nh_energy = np.full((window_size, window_size), np.nan)
                        
                        # Fill matrices with available data
                        for i, pos_i in enumerate(window_positions):
                            if pos_i in pos_to_idx:
                                idx_i = pos_to_idx[pos_i]
                                for j, pos_j in enumerate(window_positions):
                                    if pos_j in pos_to_idx:
                                        idx_j = pos_to_idx[pos_j]
                                        
                                        window_hbond_energy[i, j] = hbond_matrices['hbond_energy'][idx_i, idx_j]
                                        window_hbond_count[i, j] = hbond_matrices['hbond_count'][idx_i, idx_j]
                                        window_nh_o_energy[i, j] = hbond_matrices['nh_o_energy'][idx_i, idx_j]
                                        window_o_nh_energy[i, j] = hbond_matrices['o_nh_energy'][idx_i, idx_j]
                        
                        # Store matrices in DataFrame
                        windows_df.at[idx, 'hbond_energy_matrix'] = window_hbond_energy
                        windows_df.at[idx, 'hbond_count_matrix'] = window_hbond_count
                        windows_df.at[idx, 'nh_o_energy_matrix'] = window_nh_o_energy
                        windows_df.at[idx, 'o_nh_energy_matrix'] = window_o_nh_energy
                        
                        # Calculate mean values if requested
                        if include_hbond_mean:
                            windows_df.at[idx, 'hbond_energy_mean'] = np.nanmean(window_hbond_energy)
                            windows_df.at[idx, 'hbond_count_mean'] = np.nanmean(window_hbond_count)
                            windows_df.at[idx, 'nh_o_energy_mean'] = np.nanmean(window_nh_o_energy)
                            windows_df.at[idx, 'o_nh_energy_mean'] = np.nanmean(window_o_nh_energy)
                        
                    except Exception as matrix_error:
                        print(f"    Error processing window matrix at {center_position}: {str(matrix_error)}")
                        continue
                
            except Exception as e:
                print(f"Error processing protein {entry} for hydrogen bond analysis: {str(e)}")
                continue
        
        print(f"\nHydrogen bond analysis completed")
        
        # Report added features
        added_features = ['hbond_energy_matrix', 'hbond_count_matrix', 'nh_o_energy_matrix', 'o_nh_energy_matrix']
        if include_hbond_mean:
            added_features.extend(['hbond_energy_mean', 'hbond_count_mean', 'nh_o_energy_mean', 'o_nh_energy_mean'])
        
        print(f"Added hydrogen bond features: {added_features}")
        
        # Handle show_original_data parameter
        if not show_original_data:
            # Keep only hydrogen bond matrix columns
            columns_to_keep = ['hbond_energy_matrix', 'hbond_count_matrix', 'nh_o_energy_matrix', 'o_nh_energy_matrix']
            if include_hbond_mean:
                columns_to_keep.extend(['hbond_energy_mean', 'hbond_count_mean', 'nh_o_energy_mean', 'o_nh_energy_mean'])
            
            if columns_to_keep:
                print(f"Keeping only hydrogen bond matrix columns: {columns_to_keep}")
                windows_df = windows_df[columns_to_keep].copy()
            else:
                print("Warning: No hydrogen bond matrix columns to keep")
        else:
            print(f"Keeping all original columns plus hydrogen bond matrices")
        
        return windows_df
    

class FlexibilityProfiler(StructuralProfiler):
    def __init__(self):
        super().__init__()
        
    def profile_flexibility(self, windows_df: pd.DataFrame,
                             uniprot_id_col: str = 'entry',
                             structure_source: str = "alphafold",
                             show_original_data: bool = True,
                             include_flexibility_mean: bool = True) -> pd.DataFrame:
        """
        Profile flexibility features (pLDDT or B-factor) for all windows as vectors.
        
        Args:
            windows_df (pd.DataFrame): DataFrame with window information.
            uniprot_id_col (str): Column name for UniProt IDs.
            structure_source (str): 'alphafold' for pLDDT, 'pdb' for B-factor.
            show_original_data (bool): If True, keeps original columns.
            include_flexibility_mean (bool): If True, calculates and adds a mean flexibility column.

        Returns:
            pd.DataFrame: DataFrame with flexibility vectors and optionally means.
        """
        flexibility_col_name = "pLDDT" if structure_source == "alphafold" else "b_factor"
        
        windows_df['flexibility_vector'] = None
        if include_flexibility_mean:
            # Dynamically name the mean column based on the flexibility type
            self.flexibility_mean_col = f"{flexibility_col_name}_mean"
            windows_df[self.flexibility_mean_col] = None
        
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein for flexibility: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                flex_results = self.flexibility_analyzer.run_flexibility_analysis(
                    uniprot_id=entry,
                    structure_source=structure_source
                )

                if not flex_results.empty:
                    full_flex_dict = dict(zip(flex_results['position'], flex_results[flexibility_col_name]))
                    print(f"    Flexibility data extracted for {len(full_flex_dict)} positions.")
                else:
                    print(f"    Warning: Flexibility analysis failed or returned no data for {entry}")
                    full_flex_dict = {}

                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position']
                    window_size = row['window_size']
                    positions = self._extract_window_positions(center_position, window_size)
                    
                    if not positions:
                        continue
                    
                    flex_vector = [np.nan] * window_size
                    if full_flex_dict:
                        for i, pos in enumerate(positions):
                            if pos in full_flex_dict:
                                flex_vector[i] = full_flex_dict[pos]
                        windows_df.at[idx, 'flexibility_vector'] = flex_vector

            except Exception as e:
                print(f"Error processing protein {entry} for flexibility: {str(e)}")
                continue
        
        print(f"\nFlexibility profiling completed.")
        
        # Calculate mean flexibility if requested
        if include_flexibility_mean:
            windows_df[self.flexibility_mean_col] = windows_df['flexibility_vector'].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) and len(x) > 0 else np.nan
            )
            print(f"'{self.flexibility_mean_col}' column added.")

        # Handle show_original_data parameter
        if not show_original_data:
            columns_to_keep = ['flexibility_vector']
            if include_flexibility_mean:
                columns_to_keep.append(self.flexibility_mean_col)
            windows_df = windows_df[columns_to_keep].copy()
        
        return windows_df



class ConformationProfiler(StructuralProfiler):
    def __init__(self):
        super().__init__()
        self.feature_names = {
            'cb_cb_distance': 'CB-CB Distance (Å)',
            'omega_angle': 'Omega Angle (rad)', 
            'theta_angle': 'Theta Angle (rad)',
            'phi_angle': 'Phi Angle (rad)'
        }
        
    def _profile_conformation(self, windows_df: pd.DataFrame,
                           uniprot_id_col: str = 'entry',
                           structure_source: str = "alphafold",
                           show_original_data: bool = True) -> pd.DataFrame:
        """
        Profile conformation features for all windows as vectors
        
        For each window, generates conformation vectors where each element corresponds to
        the conformation features of that position in the window.
        
        Args:
            windows_df: DataFrame containing window information
            uniprot_id_col: Column name for UniProt IDs
            structure_source: Structure source for analysis
            show_original_data: Whether to keep the original columns
            
        Returns:
            DataFrame with conformation vectors added:
            - cb_cb_distance_vector: [d1, d2, ..., dl] CB-CB distances
            - omega_angle_vector: [ω1, ω2, ..., ωl] omega angles  
            - theta_angle_vector: [θ1, θ2, ..., θl] theta angles
            - phi_angle_vector: [φ1, φ2, ..., φl] phi angles
            - cb_cb_distance_mean: mean CB-CB distance for the window
            - omega_angle_mean: mean omega angle for the window
            - theta_angle_mean: mean theta angle for the window
            - phi_angle_mean: mean phi angle for the window
        """
        # Initialize conformation analyzer
        conformation_analyzer = ConformationAnalyzer()
        
        # Initialize conformation vector columns
        windows_df['cb_cb_distance'] = None
        windows_df['omega_angle'] = None
        windows_df['theta_angle'] = None
        windows_df['phi_angle'] = None
        
        # Process each unique protein
        for entry in windows_df[uniprot_id_col].unique():
            print(f"\nProcessing protein for conformation analysis: {entry}")
            
            entry_mask = windows_df[uniprot_id_col] == entry
            entry_windows = windows_df[entry_mask]
            
            try:
                # Get all conformation features at once
                conformation_results = conformation_analyzer.run_conformation_analysis(
                    uniprot_id=entry,
                    structure_source=structure_source
                ) # 1-based index
                if not conformation_results.empty and len(conformation_results) > 0:
                    try:
                        # Create dictionaries for different conformation metrics
                        required_cols = ['position', 'cb_cb_distance', 'omega_angle', 'theta_angle', 'phi_angle']
                        if all(col in conformation_results.columns for col in required_cols):
                            full_cb_cb_dict = dict(zip(conformation_results['position'], conformation_results['cb_cb_distance'])) # 1-based index
                            full_omega_dict = dict(zip(conformation_results['position'], conformation_results['omega_angle'])) # 1-based index
                            full_theta_dict = dict(zip(conformation_results['position'], conformation_results['theta_angle'])) # 1-based index
                            full_phi_dict = dict(zip(conformation_results['position'], conformation_results['phi_angle'])) # 1-based index
                            print(f"    Extracted conformation data for {len(full_cb_cb_dict)} positions.")
                        else:
                            missing_cols = [col for col in required_cols if col not in conformation_results.columns]
                            print(f"    Error: Missing required columns: {missing_cols}")
                            full_cb_cb_dict = {}
                            full_omega_dict = {}
                            full_theta_dict = {}
                            full_phi_dict = {}
                            
                    except Exception as conformation_error:
                        print(f"    Error extracting conformation data: {str(conformation_error)}")
                        full_cb_cb_dict = {}
                        full_omega_dict = {}
                        full_theta_dict = {}
                        full_phi_dict = {}
                else:
                    print(f"    Conformation analysis failed or returned no data for {entry}")
                    full_cb_cb_dict = {}
                    full_omega_dict = {}
                    full_theta_dict = {}
                    full_phi_dict = {}
                
                # Process each window for this protein
                for idx, row in entry_windows.iterrows():
                    center_position = row['center_position'] # 1-based index
                    window_size = row['window_size']
                    
                    # Extract window positions
                    positions = self._extract_window_positions(center_position, window_size) # 1-based index
                    interaction_sites = [pos-1 for pos in positions] # 0-based index    
                    
                    if not positions:
                        print(f"Warning: No valid positions found for window at {center_position}")
                        continue
                    
                    # Initialize vectors with NaN values
                    cb_cb_vector = [np.nan] * window_size
                    omega_vector = [np.nan] * window_size
                    theta_vector = [np.nan] * window_size
                    phi_vector = [np.nan] * window_size
                    
                    # Fill conformation vectors
                    if full_cb_cb_dict:
                        for i, pos in enumerate(positions):
                            try:
                                if pos in full_cb_cb_dict:
                                    cb_cb_vector[i] = full_cb_cb_dict[pos][interaction_sites]
                                if pos in full_omega_dict:
                                    omega_vector[i] = full_omega_dict[pos][interaction_sites]
                                if pos in full_theta_dict:
                                    theta_vector[i] = full_theta_dict[pos][interaction_sites]
                                if pos in full_phi_dict:
                                    phi_vector[i] = full_phi_dict[pos][interaction_sites]
                            except Exception as pos_error:
                                print(f"    Warning: Error processing position {pos}: {str(pos_error)}")
                                continue
                        
                        windows_df.at[idx, 'cb_cb_distance'] = cb_cb_vector
                        windows_df.at[idx, 'omega_angle'] = omega_vector
                        windows_df.at[idx, 'theta_angle'] = theta_vector
                        windows_df.at[idx, 'phi_angle'] = phi_vector
                
            except Exception as e:
                print(f"Error processing protein {entry} for conformation analysis: {str(e)}")
                continue
        
        print(f"\nConformation analysis completed")
        
        # Report added features
        added_features = ['cb_cb_distance', 'omega_angle', 'theta_angle', 'phi_angle']
        print(f"Added conformation features: {added_features}")
        
        # Handle show_original_data parameter
        if not show_original_data:
            # Keep only conformation vector columns
            columns_to_keep = ['cb_cb_distance', 'omega_angle', 'theta_angle', 'phi_angle']
            
            if columns_to_keep:
                print(f"Keeping only conformation vector columns: {columns_to_keep}")
                windows_df = windows_df[columns_to_keep].copy()
            else:
                print("Warning: No conformation vector columns to keep")
        else:
            print(f"Keeping all original columns plus conformation vectors")
        
        return windows_df
    
    def _generate_node_features(self, omega_matrix, theta_matrix, phi_matrix, window_size):
        """
        Generate node features V_i ∈ R^10 for each residue i
        V_i = {sin(·), cos(·)} × {ω_{i,i+1}, θ_{i+1,i}, θ_{i,i+1}, φ_{i+1,i}, φ_{i,i+1}}
        """
        node_features = np.zeros((window_size, 10))
        
        for i in range(window_size):
            features = []
            
            # Get angles with next residue (i+1)
            if i < window_size - 1:
                # ω_{i,i+1}: omega angle between residue i and i+1
                omega_i_ip1 = omega_matrix[i, i+1] if not np.isnan(omega_matrix[i, i+1]) else 0.0
                
                # θ_{i+1,i}: theta angle from i+1 to i
                theta_ip1_i = theta_matrix[i+1, i] if not np.isnan(theta_matrix[i+1, i]) else 0.0
                
                # θ_{i,i+1}: theta angle from i to i+1
                theta_i_ip1 = theta_matrix[i, i+1] if not np.isnan(theta_matrix[i, i+1]) else 0.0
                
                # φ_{i+1,i}: phi angle from i+1 to i
                phi_ip1_i = phi_matrix[i+1, i] if not np.isnan(phi_matrix[i+1, i]) else 0.0
                
                # φ_{i,i+1}: phi angle from i to i+1
                phi_i_ip1 = phi_matrix[i, i+1] if not np.isnan(phi_matrix[i, i+1]) else 0.0
            else:
                # For last residue, use previous residue's values
                omega_i_ip1 = omega_matrix[i-1, i] if not np.isnan(omega_matrix[i-1, i]) else 0.0
                theta_ip1_i = theta_matrix[i, i-1] if not np.isnan(theta_matrix[i, i-1]) else 0.0
                theta_i_ip1 = theta_matrix[i-1, i] if not np.isnan(theta_matrix[i-1, i]) else 0.0
                phi_ip1_i = phi_matrix[i, i-1] if not np.isnan(phi_matrix[i, i-1]) else 0.0
                phi_i_ip1 = phi_matrix[i-1, i] if not np.isnan(phi_matrix[i-1, i]) else 0.0
            
            # Apply sin/cos transformation for periodicity
            angle_values = [omega_i_ip1, theta_ip1_i, theta_i_ip1, phi_ip1_i, phi_i_ip1]
            
            for angle in angle_values:
                features.extend([np.sin(angle), np.cos(angle)])
            
            node_features[i] = features
        
        return node_features
    
    def _generate_edge_features(self, cb_matrix, omega_matrix, theta_matrix, phi_matrix, window_size):
        """
        Generate edge features E_{i,j} ∈ R^11 for each residue pair (i,j)
        E_{i,j} = {d_{i,j}} + {sin(·), cos(·)} × {ω_{i,j}, θ_{i,j}, θ_{j,i}, φ_{i,j}, φ_{j,i}}
        """
        edge_features = np.zeros((window_size, window_size, 11))
        
        for i in range(window_size):
            for j in range(window_size):
                if i == j:
                    # Self-loop: set to zeros or special values
                    edge_features[i, j] = np.zeros(11)
                    continue
                
                features = []
                
                # d_{i,j}: Euclidean distance between Cβ atoms
                distance = cb_matrix[i, j] if not np.isnan(cb_matrix[i, j]) else 0.0
                features.append(distance)
                
                # ω_{i,j}: omega angle between residues i and j
                omega_ij = omega_matrix[i, j] if not np.isnan(omega_matrix[i, j]) else 0.0
                
                # θ_{i,j}: theta angle from i to j
                theta_ij = theta_matrix[i, j] if not np.isnan(theta_matrix[i, j]) else 0.0
                
                # θ_{j,i}: theta angle from j to i
                theta_ji = theta_matrix[j, i] if not np.isnan(theta_matrix[j, i]) else 0.0
                
                # φ_{i,j}: phi angle from i to j
                phi_ij = phi_matrix[i, j] if not np.isnan(phi_matrix[i, j]) else 0.0
                
                # φ_{j,i}: phi angle from j to i
                phi_ji = phi_matrix[j, i] if not np.isnan(phi_matrix[j, i]) else 0.0
                
                # Apply sin/cos transformation for angular features
                angle_values = [omega_ij, theta_ij, theta_ji, phi_ij, phi_ji]
                
                for angle in angle_values:
                    features.extend([np.sin(angle), np.cos(angle)])
                
                edge_features[i, j] = features
        
        return edge_features
    

    def profile_conformation_features(self, windows_df: pd.DataFrame,
                                     uniprot_id_col: str = 'entry',
                                     structure_source: str = "alphafold",
                                     show_original_data: bool = True
                                     ) -> pd.DataFrame:
        """
        Generate graph neural network features for protein windows.
        
        Node features V_i ∈ R^10: {sin(·), cos(·)} × {ω_{i,i+1}, θ_{i+1,i}, θ_{i,i+1}, φ_{i+1,i}, φ_{i,i+1}}
        Edge features E_{i,j} ∈ R^11: {d_{i,j}} + {sin(·), cos(·)} × {ω_{i,j}, θ_{i,j}, θ_{j,i}, φ_{i,j}, φ_{j,i}}
        
        Args:
            windows_df: DataFrame containing window information
            uniprot_id_col: Column name for UniProt ID
            structure_source: Source of structure data
            show_original_data: Whether to include original window data
            
        Returns:
            DataFrame with node and edge features for graph neural networks
        """
        # First get the basic conformation matrices
        conformation_results = self._profile_conformation(
            windows_df, uniprot_id_col, structure_source, show_original_data=False
        )
        
        # Initialize feature columns
        conformation_results['node_features'] = None  # R^{L×10}
        conformation_results['edge_features'] = None  # R^{L×L×11}
        
        print("Generating graph neural network features...")
        
        for idx, row in conformation_results.iterrows():
            try:
                # Extract matrices
                cb_matrix = np.array(row['cb_cb_distance']) if row['cb_cb_distance'] is not None else None
                omega_matrix = np.array(row['omega_angle']) if row['omega_angle'] is not None else None
                theta_matrix = np.array(row['theta_angle']) if row['theta_angle'] is not None else None
                phi_matrix = np.array(row['phi_angle']) if row['phi_angle'] is not None else None
                
                if any(m is None for m in [cb_matrix, omega_matrix, theta_matrix, phi_matrix]):
                    print(f"Warning: Missing data for window {idx}")
                    continue
                
                window_size = cb_matrix.shape[0]
                
                # Generate Node Features V_i ∈ R^{L×10}
                node_features = self._generate_node_features(
                    omega_matrix, theta_matrix, phi_matrix, window_size
                )
                
                # Generate Edge Features E_{i,j} ∈ R^{L×L×11}
                edge_features = self._generate_edge_features(
                    cb_matrix, omega_matrix, theta_matrix, phi_matrix, window_size
                )
                
                # Store features
                conformation_results.at[idx, 'node_features'] = node_features
                conformation_results.at[idx, 'edge_features'] = edge_features
                
            except Exception as e:
                print(f"Error processing window {idx}: {str(e)}")
                continue
        
        print("Graph neural network features generated successfully!")
        
        if not show_original_data:
            # Keep only GNN-relevant columns
            gnn_columns = ['node_features', 'edge_features']
            available_cols = [col for col in gnn_columns if col in conformation_results.columns]
            if available_cols:
                conformation_results = conformation_results[available_cols]
        
        return conformation_results

