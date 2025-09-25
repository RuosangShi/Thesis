'''
Node Feature Extraction.

New Feature Dimensions (Total per residue: 2476):
- Sequence-based (148): PAM/BLOSUM (10) + AAindex (118)
- Structure-based (23): Conformation (10) + Distance (2) + RSA (1) + Secondary structure (8) + Flexibility (1) + Residue depth (1)
- PLM Embeddings (2304): ESM-2 (1280) + ProstT5 (1024)
'''

from ...AAindex_profiler import AAindex_profiler
from ...structural_analyzer.structural_profiler import *
from ...structural_analyzer.plm_profiler import ProstT5Profiler, ESM2Profiler
from ...substitution_matrix import SubstitutionMatrixProfile

from typing import List, Dict, Any
import pandas as pd
import numpy as np


class NodeFeatures:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.feature_dims = {
            'sequence': 128,  # PAM/BLOSUM (10) + AAindex (118)
            'structure': 23,  # Total structural features
            'plm': 2304  # ESM-2 (1280) + ProstT5 (1024)
        }

    def profile_features(self, 
                         include_sequence_features: bool = True,
                         include_structure_features: bool = True, 
                         include_plm_embeddings: bool = False,
                         embedding_level: str = 'residue'):
        '''
        Extract enhanced node features and return a compact structure with three keys.

        Returns:
            Dict with keys: 'sequence', 'structure', 'plm'.
            Each value is a DataFrame with columns pos_#, where each cell is a single concatenated vector for that position.
        '''
        features: Dict[str, pd.DataFrame] = {}

        # 1. SEQUENCE-BASED FEATURES (concatenate substitution_matrix + aaindex)
        if include_sequence_features:
            sequence_components = self._extract_sequence_features()
            sequence_df = self._concat_components_by_position(
                components=sequence_components,
                component_order=['substitution_matrix', 'aaindex']
            )
            features['sequence'] = sequence_df
            # Print inferred per-residue dims
            try:
                first_vec_len = len(sequence_df.iloc[0][sequence_df.columns[0]])
                print(f" Extracted sequence features (concatenated): {first_vec_len} dims per residue")
            except Exception:
                print(" Extracted sequence features (concatenated)")

        # 2. STRUCTURE-BASED FEATURES (concatenate all structure components)
        if include_structure_features:
            structure_components = self._extract_structure_features()
            structure_df = self._concat_components_by_position(
                components=structure_components,
                component_order=[
                    'conformation',
                    'sequence_distance',
                    'spatial_distance',
                    'dssp_rsa',
                    'dssp_secondary_structure',
                    'flexibility',
                    'residue_depth'
                ]
            )
            features['structure'] = structure_df
            try:
                first_vec_len = len(structure_df.iloc[0][structure_df.columns[0]])
                print(f" Extracted structure features (concatenated): {first_vec_len} dims per residue")
            except Exception:
                print(" Extracted structure features (concatenated)")

        # 3. PLM EMBEDDINGS (concatenate ProstT5 + ESM-2)
        if include_plm_embeddings:
            plm_features = self._extract_plm_features(embedding_level=embedding_level)
            plm_df = self._concat_components_by_position(
                components=plm_features,
                component_order=['prostt5', 'esm2']
            )
            features['plm'] = plm_df
            try:
                first_vec_len = len(plm_df.iloc[0][plm_df.columns[0]])
                print(f" Extracted PLM features (concatenated): {first_vec_len} dims per residue")
                print(f"    Expected: ProstT5 (1024) + ESM-2 (1280) = 2304 dims")
            except Exception:
                print(" Extracted PLM features (concatenated)")
            
        return features
    
    def _extract_sequence_features(self):
        '''Extract sequence-based features: 148 dims = PAM/BLOSUM (10) + AAindex (128) + extras (10)'''
        sequence_components: Dict[str, pd.DataFrame] = {}
        
        # PAM/BLOSUM substitution matrices (10 dims)
        substitution_matrix_profiler = SubstitutionMatrixProfile()
        print('  • Extracting PAM/BLOSUM matrices (10 dims)...')
        sequence_components['substitution_matrix'] = substitution_matrix_profiler.transform(
            self.dataframe, 
            mode='residue', 
            show_flattened_features=False,
            show_original_data=False
        )
        
        # AAindex properties (118 dims)
        aaindex_profiler = AAindex_profiler(reduce_scales=True, n_clusters=118)  # Updated to 128
        aaindex_profiler.get_aa_index()
        print('  • Extracting AAindex properties (118 dims)...')
        sequence_components['aaindex'] = aaindex_profiler.map_aa_index_to_windows(
            self.dataframe, 
            show_flattened_features=False
        )
        
        return sequence_components
    
    def _extract_structure_features(self):
        '''Extract structure-based features: 23 dims = Conformation (10) + Distance (2) + RSA (1) + Secondary structure (8) + Flexibility (1) + Residue depth (1)'''
        structure_components: Dict[str, pd.DataFrame] = {}
        
        # Conformation features (10 dims)
        conformation_profiler = ConformationProfiler()
        print('  • Extracting conformation features (10 dims)...')
        conformation_results = conformation_profiler.profile_conformation_features(
            self.dataframe,
            uniprot_id_col='entry',
            structure_source="alphafold",
            show_original_data=False
        )
        conformation_node_series = conformation_results['node_features'].apply(
                lambda arr: pd.Series(
                    [arr[i].tolist() for i in range(arr.shape[0])], 
                    index=[f'pos_{i+1}' for i in range(arr.shape[0])]
                )
            )    
        structure_components['conformation'] = conformation_node_series


        # Distance features (2 dims: sequence + spatial)
        distance_profiler = DistanceProfiler()
        print('  • Extracting distance features (2 dims)...')
        distance_results = distance_profiler.profile_distance(
            self.dataframe,
            include_sequence_distance=True,
            include_spatial_distance=True,
            structure_source="alphafold",
            show_original_data=False,
            only_extracellular_sites=False,
            include_sequence_distance_mean=False,
            include_spatial_distance_mean=False
        )
        structure_components['sequence_distance'] = distance_results['sequence_distance_vector'].apply(
                                            lambda arr: pd.Series(arr, index=[f'pos_{i+1}' for i in range(len(arr))])
                                            )
        structure_components['spatial_distance'] = distance_results['spatial_distance_vector'].apply(
                                            lambda arr: pd.Series(arr, index=[f'pos_{i+1}' for i in range(len(arr))])
                                            )
        
        # DSSP features: RSA (1 dim) + Secondary structure (9 dims) 
        dssp_profiler = DSSPProfiler()
        print('  • Extracting DSSP features - RSA (1 dim) + Secondary structure (9 dims)...')
        dssp_results = dssp_profiler.profile_dssp(
            self.dataframe,
            include_rsa=True,
            include_secondary_structure=True,
            rsa_cal='Wilke',
            structure_source="alphafold",
            show_original_data=False,
            include_rsa_mean=False
        )
        structure_components['dssp_rsa'] = dssp_results['rsa_vector'].apply(
                lambda arr: pd.Series(arr, index=[f'pos_{i+1}' for i in range(len(arr))])
            )
        def encode_secondary_structure(arr):
            # Define secondary structure categories
            ss_categories = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']  # 8 DSSP categories
                
            # Create one-hot encoding for each position
            encoded_positions = []
            for i, ss_char in enumerate(arr):
                # Create one-hot vector for this position
                one_hot = [1 if ss_char == cat else 0 for cat in ss_categories]
                encoded_positions.append(one_hot)
                
            # Convert to pandas Series with multi-level indexing
            return pd.Series(encoded_positions, index=[f'pos_{i+1}' for i in range(len(arr))])
            
        structure_components['dssp_secondary_structure'] = dssp_results['secondary_structure_vector'].apply(encode_secondary_structure)

        
        # Flexibility features (1 dim)
        flexibility_profiler = FlexibilityProfiler()
        print('  • Extracting flexibility features (1 dim)...')
        flexibility_results = flexibility_profiler.profile_flexibility(
            self.dataframe,
            include_flexibility_mean=False,
            show_original_data=False,
            structure_source="alphafold"
        )
        structure_components['flexibility'] = flexibility_results['flexibility_vector'].apply(
                                      lambda arr: pd.Series(arr, index=[f'pos_{i+1}' for i in range(len(arr))])
                                      )
        
        # Residue depth features (1 dim)
        residue_depth_profiler = ResidueDepthProfiler()
        print('  • Extracting residue depth features (1 dim)...')
        residue_depth_results = residue_depth_profiler.profile_residue_depth(
            self.dataframe,
            uniprot_id_col='entry', 
            structure_source='alphafold', 
            show_original_data=False, 
            include_depth_mean=False
        )
        structure_components['residue_depth'] = residue_depth_results['residue_depth_vector'].apply(
                lambda arr: pd.Series(arr, index=[f'pos_{i+1}' for i in range(len(arr))])
            )
        
        return structure_components
    
    def _extract_plm_features(self, embedding_level='residue'):
        '''Extract PLM features: 2304 dims = ESM-2 (1280) + ProstT5 (1024)'''
        print('  • Extracting PLM embeddings...')
        
        plm_results: Dict[str, pd.DataFrame] = {}
        
        # Extract ProstT5 embeddings (1024 dims)
        print('    - ProstT5 embeddings (1024 dims)...')
        prostt5_profiler = ProstT5Profiler(
            use_disk_cache=True,
            use_ram_cache=True
        )
        plm_results['prostt5'] = prostt5_profiler.profile_plm(
            windows_df=self.dataframe,
            embedding_level=embedding_level,
            show_original_data=False
        )
        
        # Extract ESM-2 embeddings (1280 dims)
        print('    - ESM-2 embeddings (1280 dims)...')
        esm2_profiler = ESM2Profiler(
                use_disk_cache=True,
                use_ram_cache=True
        )
        plm_results['esm2'] = esm2_profiler.profile_plm(
                    windows_df=self.dataframe,
                    embedding_level=embedding_level,
                show_original_data=False
        )
        
        return plm_results
    
    def _create_esm2_placeholder(self):
        '''Create ESM-2 placeholder embeddings that match ProstT5 format'''
        esm2_embedding_data = []
        for index, row in self.dataframe.iterrows():
            sequence = row['sequence']
            window_size = len(sequence)
            
            # Generate random embeddings for each position in the window
            embeddings = np.random.randn(window_size, 1280)  # Shape: (window_size, 1280)
            
            # Create row data with pos_1, pos_2, etc. columns, each storing a 1280-dim vector
            row_data = {}
            for pos_idx in range(embeddings.shape[0]):
                col_name = f'pos_{pos_idx + 1}'  # pos_1, pos_2, etc.
                row_data[col_name] = embeddings[pos_idx]  # Store 1280-dim vector
            esm2_embedding_data.append(row_data)
        
        # Create DataFrame from embedding data
        return pd.DataFrame(esm2_embedding_data, index=self.dataframe.index)


    def _concat_components_by_position(self, components: Dict[str, pd.DataFrame], component_order: List[str]) -> pd.DataFrame:
        '''
        Concatenate multiple component DataFrames per position into a single per-position vector.
        - components: dict mapping component name -> DataFrame (columns: pos_#; cells: array-like or scalar)
        - component_order: explicit order to concatenate components
        Returns a DataFrame (rows = input rows, columns = pos_#) with each cell being a 1D numpy array.
        '''
        if not components:
            return pd.DataFrame(index=self.dataframe.index)

        # Use columns from the first component in the provided order; enforce consistent ordering by numeric position
        sample_component = components[component_order[0]]
        pos_columns = sorted(sample_component.columns, key=lambda c: int(str(c).split('_')[-1]))

        combined_rows = []
        for idx in self.dataframe.index:
            pos_to_vec: Dict[str, np.ndarray] = {}
            for pos_col in pos_columns:
                parts: List[np.ndarray] = []
                for comp_name in component_order:
                    comp_df = components.get(comp_name)
                    if comp_df is None:
                        continue
                    value = comp_df.loc[idx, pos_col]
                    # Normalize value to 1D numpy array
                    arr = value
                    if isinstance(value, (list, tuple)):
                        arr = np.array(value)
                    elif not isinstance(value, np.ndarray):
                        # scalar -> array
                        arr = np.array([value])
                    arr = np.ravel(arr)
                    parts.append(arr)
                if parts:
                    pos_to_vec[pos_col] = np.concatenate(parts, axis=0)
                else:
                    pos_to_vec[pos_col] = np.array([])
            combined_rows.append(pos_to_vec)
        combined_df = pd.DataFrame(combined_rows, index=self.dataframe.index)
        return combined_df

