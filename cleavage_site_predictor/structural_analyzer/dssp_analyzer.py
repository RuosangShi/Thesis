'''
This module contains the DSSPAnalysis class
It uses the DSSP algorithm to calculate the secondary structure and RSA of a protein.
'''

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, DSSP
from typing import List, Dict, Tuple
import os

from .structure_paser import StructureParser
from .structure_downloader import StructureDownloader
from .structure_downloader import StructureFormatter
from .cache_manager import CacheManager


class DSSPAnalysis(CacheManager): 
    DSSP_LABELS = {
        '-': 'Loop/irregular', 'B': 'β-bridge', 'E': 'β-strand',
        'G': '3-10 helix', 'H': 'α-helix', 'I': 'π-helix',
        'P': 'κ-helix', 'S': 'Bend', 'T': 'Turn'
    }
    
    def __init__(self, cache_dir: str = "structure_temp/dssp_cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = True):
        """Initialize DSSP Analysis with dual caching system (RAM + Disk)."""
        # call parent class to initialize cache system
        super().__init__(cache_dir=cache_dir, 
                        use_disk_cache=use_disk_cache, 
                        use_ram_cache=use_ram_cache)
        
        # initialize DSSP analysis components
        self.downloader = StructureDownloader()
        self.formatter = StructureFormatter()
        self.parser = StructureParser()
        
    '''Manage cache'''
    def _get_dssp_cache_key(self, uniprot_id: str, pdb_id: str = None, 
                           structure_source: str = "alphafold", 
                           rsa_cal: str = None) -> str:
        """generate DSSP cache key"""
        return self._generate_cache_key(
            uniprot_id=uniprot_id,
            pdb_id=pdb_id, 
            structure_source=structure_source,
            rsa_cal=rsa_cal
        )
        
    def _save_cache_to_disk(self, cache_key: str, data: dict, cache_type: str):
        """save cache to disk"""
        return self._save_to_disk(cache_key, data, cache_type)
            
    def _load_cache_from_disk(self, cache_key: str, cache_type: str) -> dict:
        """load cache from disk"""
        return self._load_from_disk(cache_key, cache_type)
        
    def _get_or_create_pdb_cache(self, uniprot_id: str, pdb_id: str = None, 
                                structure_source: str = "alphafold") -> Tuple[str, str]:
        """get or create PDB cache (using parent class's two-level cache system)"""
        cache_key = self._get_dssp_cache_key(uniprot_id, pdb_id, structure_source)
        
        # try to get data from two-level cache
        cached_data = self._get_cached_data(cache_key, "pdb")
        if cached_data and os.path.exists(cached_data.get('filtered_pdb_path', '')):
            return (cached_data['filtered_pdb_path'], cached_data['selected_chain'])
            
        # create new PDB cache
        print(f" create new PDB cache: {uniprot_id}")
        valid_chains, valid_pdb = self.downloader.read_pdb_by_source(
            structure_source=structure_source, 
            uniprot_id=uniprot_id, pdb_id=pdb_id, chain=None)
            
        if not valid_chains:
            return None, None
            
        # use the first valid chain
        selected_chain = list(valid_chains)[0]
        temp_pdb = self.parser._filter_pdb_by_chain(valid_pdb, selected_chain)    
        filtered_pdb_path = self.parser._save_filtered_pdb(temp_pdb)
        
        # prepare cache data
        cache_data = {
            'filtered_pdb_path': filtered_pdb_path,
            'selected_chain': selected_chain
        }
        
        # save to two-level cache
        self._cache_data(cache_key, cache_data, "pdb")
        
        return (filtered_pdb_path, selected_chain)


    '''get DSSP results for all residues in the sequence'''
    def _calculate_full_dssp_properties(self, pdb_path, rsa_cal: str = 'Wilke'):
        """Calculate RSA and secondary structure for all residues in the sequence."""

        adjusted_file_path = f"{pdb_path.rsplit('.', 1)[0]}_adjusted.pdb"
        self.formatter.format_pdb(pdb_path, adjusted_file_path)

        try:
            structure = PDBParser().get_structure('protein', adjusted_file_path)
            # Run DSSP on the structure once
            dssp = DSSP(structure[0], adjusted_file_path, acc_array=rsa_cal)
            # Extract properties into a dictionary
            dssp_properties = {}
            for key in dssp.keys():
                residue_num = key[1][1]
                dssp_data = dssp[key]
                aa = dssp_data[1] 
                ss = dssp_data[2] # Secondary structure
                rsa = dssp_data[3] # Relative Solvent Accessibility
                phi = dssp_data[4] # Phi angle
                psi = dssp_data[5] # Psi angle
                NH_O_1_relidx = dssp_data[6] # Relative index of acceptor residue for first H-bond (N-H → O)
                NH_O_1_energy = dssp_data[7] # Energy (kcal/mol) of that H-bond
                O_NH_1_relidx = dssp_data[8] # Relative index of donor residue for first H-bond (O → N-H)
                O_NH_1_energy = dssp_data[9] # Energy of that reverse H-bond
                NH_O_2_relidx = dssp_data[10] # Same as above, but for second N-H → O H-bond
                NH_O_2_energy = dssp_data[11] # Energy of that bond
                O_NH_2_relidx = dssp_data[12] # Same as above, but for second O → N-H H-bond
                O_NH_2_energy = dssp_data[13] # Energy of that bond

                dssp_properties[residue_num] = {'aa': aa, 'ss': ss, 'rsa': rsa, 'phi': phi, 'psi': psi,
                                                'NH_O_1_relidx': NH_O_1_relidx, 'NH_O_1_energy': NH_O_1_energy,
                                                'O_NH_1_relidx': O_NH_1_relidx, 'O_NH_1_energy': O_NH_1_energy,
                                                'NH_O_2_relidx': NH_O_2_relidx, 'NH_O_2_energy': NH_O_2_energy,
                                                'O_NH_2_relidx': O_NH_2_relidx, 'O_NH_2_energy': O_NH_2_energy}

            return dssp_properties

        except Exception as e:
            print(f"Warning: DSSP calculation failed - {str(e)}")
            return {}

    '''get or create DSSP cache for all residues in the sequence'''
    def _get_or_create_full_dssp_cache(self, uniprot_id: str, pdb_id: str = None, 
                                 structure_source: str = "alphafold", 
                                 rsa_cal: str = 'Wilke') -> Dict:
        """Get or create full DSSP analysis cache (RSA + SS) using parent class's two-level cache system."""
        cache_key = self._get_dssp_cache_key(uniprot_id, pdb_id, structure_source, rsa_cal)
        
        # try to get data from two-level cache
        cached_data = self._get_cached_data(cache_key, "dssp_full_analysis")
        if cached_data is not None:
            return cached_data
            
        # create new DSSP cache
        print(f" calculate new DSSP cache: {uniprot_id} (RSA cal: {rsa_cal})")
        filtered_pdb_path, selected_chain = self._get_or_create_pdb_cache(
            uniprot_id, pdb_id, structure_source)
            
        if not filtered_pdb_path:
            return {}
            
        dssp_properties = self._calculate_full_dssp_properties(filtered_pdb_path, rsa_cal)
        
        # save to two-level cache
        self._cache_data(cache_key, dssp_properties, "dssp_full_analysis")
        
        return dssp_properties



    def run_dssp_analysis(self, uniprot_id: str, 
                          sites: List[int] = None,
                          pdb_id: str = None, rsa_cal: str = 'Wilke', 
                          structure_source: str = "alphafold") -> pd.DataFrame:
        """
        This function calculates RSA and secondary structure for specified sites or the full sequence.
        
        Args:
            uniprot_id: UniProt ID
            sites: Optional list of sites to analyze. If None, analyzes the full sequence.
            pdb_id: Optional, specify the PDB ID
            rsa_cal: RSA calculation method ('Wilke', 'Sander', 'Miller')
            structure_source: Structure source ("pdb" or "alphafold")
            
        Returns:
            A DataFrame containing DSSP data (RSA, SS) for specified sites or all residues.
        """
        
        full_dssp_props = self._get_or_create_full_dssp_cache(
            uniprot_id=uniprot_id, 
            pdb_id=pdb_id, 
            structure_source=structure_source, 
            rsa_cal=rsa_cal
        )
        
        if not full_dssp_props:
            print("Failed to get DSSP properties")
            return pd.DataFrame()
        
        # Get chain information (from PDB cache)
        _, selected_chain = self._get_or_create_pdb_cache(uniprot_id, pdb_id, structure_source)
        
        # Determine which positions to include in the output
        positions_to_process = sites if sites is not None else sorted(full_dssp_props.keys())
        print(positions_to_process)
        results_list = []
        for pos in positions_to_process:
            if pos in full_dssp_props:
                props = full_dssp_props[pos]
                results_list.append({
                    'position': pos,
                    'aa': props['aa'],
                    'chain': selected_chain,
                    'rsa': props['rsa'],
                    'reference': rsa_cal,
                    'dssp': props['ss'],
                    'dssp_label': self.DSSP_LABELS.get(props['ss'], 'Unknown'),
                    'phi': props['phi'],
                    'psi': props['psi'],
                    'NH_O_1_relidx': props['NH_O_1_relidx'],
                    'NH_O_1_energy': props['NH_O_1_energy'],
                    'O_NH_1_relidx': props['O_NH_1_relidx'],
                    'O_NH_1_energy': props['O_NH_1_energy'],
                    'NH_O_2_relidx': props['NH_O_2_relidx'],
                    'NH_O_2_energy': props['NH_O_2_energy'],
                    'O_NH_2_relidx': props['O_NH_2_relidx'],
                    'O_NH_2_energy': props['O_NH_2_energy'],
                })
            else:
                print(f'Site {pos} cannot be found in the sequence')

        if not results_list:
            print(f"No valid DSSP data found for the requested positions.")
            return pd.DataFrame()
        
        # Rename 'position' column to 'site' if specific sites were requested
        df = pd.DataFrame(results_list)
        if sites is not None:
            df = df.rename(columns={'position': 'site'})
            
        return df
    

    def generate_hbond_matrices(self, dssp_properties: Dict[str, Dict],
                                threshold: float = None) -> Dict[str, np.ndarray]:
        '''
        Generate hydrogen bond matrices from DSSP results.
        
        Args:
            dssp_properties: Dictionary containing DSSP properties for each residue
            
        Returns:
            Dictionary containing hydrogen bond matrices:
            - 'hbond_energy': Matrix of hydrogen bond energies
            - 'hbond_count': Matrix of hydrogen bond counts
            - 'nh_o_energy': Matrix of N-H → O hydrogen bond energies
            - 'o_nh_energy': Matrix of O → N-H hydrogen bond energies
        '''
        if not dssp_properties:
            return {}
            
        # Get sorted residue positions
        positions = sorted(dssp_properties.keys()) # [1, 2, 3, ...] 1-based index
        n_residues = len(positions)
        
        # Create position to index mapping
        pos_to_idx = {pos: idx for idx, pos in enumerate(positions)} # {1:0, 2:1, 3:2, ...} 
        # Initialize matrices
        hbond_energy = np.zeros((n_residues, n_residues))
        hbond_count = np.zeros((n_residues, n_residues))
        nh_o_energy = np.zeros((n_residues, n_residues))
        o_nh_energy = np.zeros((n_residues, n_residues))
        
        # Process each residue
        for i, pos in enumerate(positions): # i: 0 based index, pos: 1-based index
            props = dssp_properties[pos]
            
            # Process N-H → O hydrogen bonds (NH_O_1, NH_O_2)
            for bond_suffix in ['1', '2']:
                relidx_key = f'NH_O_{bond_suffix}_relidx'
                energy_key = f'NH_O_{bond_suffix}_energy'
                
                if relidx_key in props and energy_key in props:
                    rel_idx = props[relidx_key]
                    energy = props[energy_key]
                    # Calculate absolute position of bonded residue
                    if rel_idx != 0 and energy != 0.0:
                        bonded_pos = pos + rel_idx # 1-based index
                        # Check if bonded position exists in our dataset
                        j = pos_to_idx[bonded_pos] # 0 based index
                            
                        # Record the hydrogen bond
                        hbond_energy[i, j] += abs(energy)  # Use absolute value for energy
                        hbond_count[i, j] += 1
                        nh_o_energy[i, j] += abs(energy)
            
            # Process O → N-H hydrogen bonds (O_NH_1, O_NH_2)
            for bond_suffix in ['1', '2']:
                relidx_key = f'O_NH_{bond_suffix}_relidx'
                energy_key = f'O_NH_{bond_suffix}_energy'
                
                if relidx_key in props and energy_key in props:
                    rel_idx = props[relidx_key]
                    energy = props[energy_key]
                    # Calculate absolute position of bonded residue
                    if rel_idx != 0 and energy != 0.0:
                        bonded_pos = pos + rel_idx
                        # Check if bonded position exists in our dataset
                        j = pos_to_idx[bonded_pos]
                            
                        # Record the hydrogen bond
                        hbond_energy[i, j] += abs(energy)  # Use absolute value for energy
                        hbond_count[i, j] += 1
                        o_nh_energy[i, j] += abs(energy)
        if threshold is not None:
            # Filter out hydrogen bonds below threshold
            hbond_energy[hbond_energy < threshold] = 0
            nh_o_energy[nh_o_energy < threshold] = 0
            o_nh_energy[o_nh_energy < threshold] = 0
        
        return {
            'hbond_energy': hbond_energy,
            'hbond_count': hbond_count,
            'nh_o_energy': nh_o_energy,
            'o_nh_energy': o_nh_energy
        }
    
        
    def visualize_hbond_matrix(self, hbond_matrix: np.ndarray, 
                              cleavage_site_position: int = None,
                              matrix_type: str = 'hbond_energy',
                              title: str = None,
                              label_name: str = 'H-bond Energy (kcal/mol)',
                              figsize: tuple = (12, 10),
                              show_residue_labels: bool = True,
                              label_interval: int = None) -> None:
        """
        Visualize hydrogen bond matrix with cleavage site position marked.
        
        Args:
            hbond_matrix: Hydrogen bond matrix (window_size × window_size)
            cleavage_site_position: Cleavage site position in the window (1-based)
            matrix_type: Matrix type, used for title display
            title: Custom title
            figsize: Figure size
            show_residue_labels: Whether to show residue labels
            label_interval: Interval for showing residue labels (auto if None)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        if hbond_matrix is None or not isinstance(hbond_matrix, np.ndarray):
            print("Warning: Invalid hydrogen bond matrix data")
            return
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        colors = ['white', '#f0f8ff', '#b3d9ff', '#66c2ff', '#1a8cff', '#0066cc', '#004499', '#002266', '#001133']
        n_bins = 64
        cmap = LinearSegmentedColormap.from_list('ultra_contrast_blue', colors, N=n_bins)
        
        # Create heatmap with ultra-enhanced contrast
        # Set more aggressive vmin/vmax for better contrast
        non_zero_values = hbond_matrix[hbond_matrix > 0]
        if len(non_zero_values) > 0:
            vmin = 0  # Always start from 0 for maximum contrast
            vmax = np.nanpercentile(non_zero_values, 95)  # Use 95th percentile to avoid outliers
            # If the range is too small, expand it
            if vmax - vmin < 0.1:
                vmax = vmin + 0.1
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(hbond_matrix, cmap=cmap, aspect='auto', origin='lower', 
                      vmin=vmin, vmax=vmax, alpha=0.9)
        
        
        # Add color bar with better styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
        cbar.set_label(label_name, fontweight='bold', fontsize=12)
        
        # Set axis labels
        ax.set_xlabel('Residue Number', fontweight='bold', fontsize=12)
        ax.set_ylabel('Residue Number', fontweight='bold', fontsize=12)
        
        # Set title
        if title is None:
            matrix_name = matrix_type.replace('_', ' ').title()
            title = f'{matrix_name} Matrix (Residue vs Residue)'
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
        
        # Set residue labels with smart interval
        if show_residue_labels:
            window_size = hbond_matrix.shape[0]
            residue_positions = list(range(1, 1 + window_size))
            
            # Auto-determine label interval to avoid crowding
            if label_interval is None:
                if window_size <= 10:
                    label_interval = 1  # Show all labels for small matrices
                elif window_size <= 20:
                    label_interval = 2  # Show every 2nd label
                elif window_size <= 50:
                    label_interval = 5  # Show every 5th label
                else:
                    label_interval = 10  # Show every 10th label for large matrices
            
            # Create tick positions and labels
            tick_positions = list(range(0, window_size, label_interval))
            tick_labels = [residue_positions[i] for i in tick_positions]
            
            # Also include first and last positions if not already included
            if 0 not in tick_positions:
                tick_positions.insert(0, 0)
                tick_labels.insert(0, residue_positions[0])
            if window_size - 1 not in tick_positions:
                tick_positions.append(window_size - 1)
                tick_labels.append(residue_positions[-1])
            
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)
            ax.set_yticklabels(tick_labels, fontsize=10)
        else:
            # Remove tick labels but keep ticks
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Mark cleavage site with enhanced visibility
        if cleavage_site_position is not None:
            # Convert to matrix index (0-based)
            cleavage_idx = cleavage_site_position - 1
            
            if 0 <= cleavage_idx < hbond_matrix.shape[0]:
                # Draw high-contrast cross at cleavage site
                ax.axhline(y=cleavage_idx, color='red', linestyle='-', linewidth=3, alpha=0.8)
                ax.axvline(x=cleavage_idx, color='red', linestyle='-', linewidth=3, alpha=0.8)
                
                # Add prominent cleavage site marker
                ax.plot(cleavage_idx, cleavage_idx, marker='*', color='red', markersize=20, 
                       markeredgecolor='white', markeredgewidth=2, 
                       label=f'Cleavage Site (Pos {cleavage_site_position})')
                
                # Add legend with better styling
                legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11,
                                 frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
        
        # Add subtle grid for better readability
        ax.set_xticks(np.arange(-0.5, hbond_matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, hbond_matrix.shape[0], 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.6)
        
        # Set border color
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        plt.tight_layout()
        plt.show()


