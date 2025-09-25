"""
Sliding window slicer - responsible for extracting training windows and generating negative samples
Support region restriction and structural distance constraint window generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from .structural_analyzer.distance_analyzer import DistanceAnalysis
import aaanalysis as aa
import os
import hashlib

class WindowSlicer:
    """sliding window slicer (support region restriction and structural distance constraint)"""
    
    def __init__(self, window_sizes: List[int] = [20], uniprot_timeout: int = 30, 
                 add_cache: bool = True, cache_dir: str = "./uniprot_cache"):
        """
        Initialize the window slicer
        
        Args:
            window_sizes: Size of the sliding window, default is [10, 14]
            uniprot_timeout: Timeout for UniProt API requests (seconds)
            add_cache: Whether to use disk cache for sequences (default: True)
            cache_dir: Directory to store cache files (default: './uniprot_cache')
        """
        self.window_sizes = window_sizes
        self.uniprot_timeout = uniprot_timeout
        self.add_cache = add_cache
        self.cache_dir = cache_dir
        
        # Create cache directory if using cache
        if self.add_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize components
        self.distance_analyzer = DistanceAnalysis()
        
        print(f"WindowSlicer initialized:")
        print(f"  Window sizes: {self.window_sizes}")
        print(f"  UniProt timeout: {self.uniprot_timeout}s")
        print(f"  Cache enabled: {self.add_cache}")
        if self.add_cache:
            print(f"  Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, uniprot_id: str, data_type: str = "sequence") -> str:
        """
        Generate cache key for the given UniProt ID and data type
        
        Args:
            uniprot_id: UniProt ID
            data_type: Type of data being cached (e.g., 'sequence')
            
        Returns:
            str: Hash-based cache key
        """
        cache_string = f"{uniprot_id}_{data_type}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """
        Get full path for cache file
        
        Args:
            cache_key: Cache key for the file
            
        Returns:
            str: Full path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.txt")
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Load data from cache file
        
        Args:
            cache_key: Cache key for the file
            
        Returns:
            str: Cached data if exists, None otherwise
        """
        if not self.add_cache:
            return None
            
        cache_file = self._get_cache_file_path(cache_key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Failed to read cache file {cache_file}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: str) -> None:
        """
        Save data to cache file
        
        Args:
            cache_key: Cache key for the file
            data: Data to be cached
        """
        if not self.add_cache:
            return
            
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(data)
        except Exception as e:
            print(f"Warning: Failed to write cache file {cache_file}: {e}")
    
    def _get_sequence_for_uniprot(self, uniprot_id: str) -> str:
        """
        Get the protein sequence for a given UniProt ID with caching support
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            str: Protein sequence
        """
        import requests
        
        # Generate cache key
        cache_key = self._get_cache_key(uniprot_id, "sequence")
        
        # Try to load from cache first
        cached_sequence = self._load_from_cache(cache_key)
        if cached_sequence is not None:
            print(f"✓ Loaded sequence for {uniprot_id} from cache (length: {len(cached_sequence)})")
            return cached_sequence
        
        # Fetch from UniProt API
        print(f"Fetching sequence for {uniprot_id} from UniProt...")
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        
        try:
            response = requests.get(url, timeout=self.uniprot_timeout)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])  # skip the FASTA header
                
                # Save to cache if caching is enabled
                self._save_to_cache(cache_key, sequence)
                if self.add_cache:
                    print(f"✓ Cached sequence for {uniprot_id}")
                
                print(f"Downloaded sequence for {uniprot_id} (length: {len(sequence)})")
                return sequence
            else:
                raise ValueError(f"Cannot get the sequence for UniProt ID {uniprot_id}, HTTP status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise ValueError(f"Timeout while downloading sequence for UniProt ID {uniprot_id} (timeout: {self.uniprot_timeout}s)")
        except requests.exceptions.ConnectionError as e:
            raise ValueError(f"Connection error while downloading sequence for UniProt ID {uniprot_id}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error while downloading sequence for UniProt ID {uniprot_id}: {str(e)}")
    
    def _filter_residues_by_distance(self, 
                                   uniprot_id: str,
                                   extracellular_regions: List[Tuple[int, int]],
                                   transmembrane_regions: List[Tuple[int, int]],
                                   intracellular_regions: List[Tuple[int, int]] = None,
                                   sequence_length: int = None,
                                   enable_distance_filter: bool = True,
                                   max_sequence_distance: int = 150,
                                   max_spatial_distance: float = 60.0,
                                   structure_source: str = "alphafold") -> List[int]:
        """
        Filter residues within extracellular regions based on distance constraints (optional)
        
        Args:
            uniprot_id: UniProt ID
            extracellular_regions: Extracellular regions list
            transmembrane_regions: Transmembrane regions list
            intracellular_regions: Intracellular regions list (optional)
            sequence_length: Length of the protein sequence
            enable_distance_filter: Whether to enable distance filtering
            max_sequence_distance: Maximum sequence distance (amino acids)
            max_spatial_distance: Maximum spatial distance (Å)
            structure_source: Structure source for spatial analysis
            
        Returns:
            List[int]: List of residues within the extracellular regions, 
            optionally filtered by sequence and spatial distance
        """
        # get all residues within the extracellular regions
        all_extracellular_sites = []
        
        # If extracellular regions are provided, use them
        if extracellular_regions:
            for start, end in extracellular_regions:
                # ensure start <= end
                min_pos, max_pos = min(start, end), max(start, end)
                all_extracellular_sites.extend(range(min_pos, max_pos + 1))
            print(f"Using provided extracellular regions: {len(all_extracellular_sites)} residues")
        
        # If no extracellular regions provided, infer from non-transmembrane and non-intracellular regions
        else:
            if sequence_length is None:
                print("Warning: Cannot infer extracellular regions without sequence length")
                return []
            
            print("No extracellular regions provided, inferring from non-transmembrane and non-intracellular regions")
            
            # Use distance_analyzer's method to get extracellular regions
            extracellular_regions = self.distance_analyzer.get_extracellular_regions(
                sequence_length=sequence_length,
                intracellular_regions=intracellular_regions,
                transmembrane_regions=transmembrane_regions
            )
            
            # Convert regions to sites list
            for start, end in extracellular_regions:
                min_pos, max_pos = min(start, end), max(start, end)
                all_extracellular_sites.extend(range(min_pos, max_pos + 1))
            
            print(f"Inferred extracellular regions: {len(all_extracellular_sites)} residues")
            print(f"  Total sequence length: {sequence_length}")
            print(f"  Excluded (TM + IC): {sequence_length - len(all_extracellular_sites)} residues")
        
        if not all_extracellular_sites:
            print("No extracellular sites found")
            return []
        
        # If distance filtering is disabled, return all extracellular sites
        if not enable_distance_filter:
            print(f"Distance filtering disabled - using all {len(all_extracellular_sites)} extracellular residues")
            return sorted(all_extracellular_sites)
        
        # Distance filtering is enabled - proceed with filtering
        print(f"Distance filtering enabled - filtering {len(all_extracellular_sites)} extracellular residues")
        
        # sequence distance filtering
        sequence_results = self.distance_analyzer.distance_analysis(
            sites=all_extracellular_sites,
            distance_type="sequence",
            extracellular_regions=extracellular_regions,
            transmembrane_regions=transmembrane_regions,
            only_extracellular_sites=True
        )
        
        sequence_filtered = []
        if not sequence_results.empty:
            sequence_filtered = sequence_results[
                sequence_results['distance'] <= max_sequence_distance
            ]['site'].tolist()
        # spatial distance filtering (if structure can be obtained)
        spatial_filtered = []
        try:
            spatial_results = self.distance_analyzer.distance_analysis(
                sites=all_extracellular_sites,
                uniprot_id=uniprot_id,
                distance_type="coordination",
                extracellular_regions=extracellular_regions,
                transmembrane_regions=transmembrane_regions,
                structure_source=structure_source,
                only_extracellular_sites=True
            )

            if not spatial_results.empty:
                spatial_filtered = spatial_results[
                    spatial_results['distance'] <= max_spatial_distance
                ]['site'].tolist()
                
        except Exception as e:
            print(f"Warning: Cannot perform spatial distance filtering for {uniprot_id}: {str(e)}")
            spatial_filtered = []
        
        # merge two filtering results (take intersection)
        if spatial_filtered:
            # Use intersection - only sites that pass BOTH sequence AND spatial filtering
            filtered_sites = list(set(sequence_filtered) & set(spatial_filtered))
            print(f"Sequence filtered: {len(sequence_filtered)} sites")
            print(f"Spatial filtered: {len(spatial_filtered)} sites")
            print(f"Using intersection of sequence and spatial filtering: {len(filtered_sites)} sites")
        else:
            # If no spatial filtering available, use only sequence filtering
            filtered_sites = sequence_filtered
            print(f"No spatial filtering available, using only sequence filtering: {len(filtered_sites)} sites")
            
        print(f"After distance filtering: {len(filtered_sites)} residues remain")
        return sorted(filtered_sites)
    
    def _create_window(self, sequence: str, center_pos: int, window_size: int, padding_char: str = 'X') -> str:
        """
        Create a window centered on the specified position
        
        Args:
            sequence: Protein sequence
            center_pos: Center position (1-based)
            window_size: Window size
            padding_char: Padding character

            Example:
            sequence = 'MKLVFF', window_size = 10, center_pos = 3, padding_char = 'X'
            -> center_pos=3 corresponds to 'L' (1-based indexing)
            -> window will be: 'XXMKLVFFXX' (padded with X on both sides)
            
        Returns:
            str: Window sequence
        """
        # convert to 0-based index
        center_idx = center_pos - 1 
        half_window = window_size // 2
        
        # calculate the start and end positions of the window
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window + 1
        
        # if the window size is even, adjust the window
        if window_size % 2 == 0:
            end_idx = center_idx + half_window
        
        # extract the window sequence
        window_seq = ""
        
        for i in range(start_idx, end_idx):
            if i < 0:
                # left padding
                window_seq += padding_char
            elif i >= len(sequence):
                # right padding
                window_seq += padding_char
            else:
                window_seq += sequence[i]
        
        return window_seq

    def _cd_hit_clustering(self, windows_df: pd.DataFrame, threshold: float = 0.4, entry_col: str = 'entry', sequence_col: str = 'sequence') -> pd.DataFrame:
        """
        Perform CD-HIT clustering on the windows, applied separately for each sequence length.
        
        Strategy:
        1. Divide dataframe by sequence length (window_size column)
        2. Apply CD-HIT clustering within each group
        3. Concatenate filtered results back together
        """
        if sequence_col not in windows_df.columns or 'window_size' not in windows_df.columns:
            print(f"Warning: Required columns not found in DataFrame")
            return windows_df
        
        print(f"Starting CD-HIT clustering with threshold {threshold}")
        print(f"Original dataset: {len(windows_df)} windows")
        
        # Group by window_size (which corresponds to sequence length)
        unique_window_sizes = sorted(windows_df['window_size'].unique())
        print(f"Window sizes found: {unique_window_sizes}")
        
        filtered_subsets = []
        
        for window_size in unique_window_sizes:
            # Extract subset for this window size
            subset_df = windows_df[windows_df['window_size'] == window_size].copy().reset_index(drop=True)
            print(f"Window size {window_size}: {len(subset_df)} sequences, applying CD-HIT clustering")

            # Prepare data for CD-HIT (need entry and sequence columns with clean indices)
            cdhit_input = subset_df[[entry_col, sequence_col]]
            try:
                # Apply CD-HIT clustering
                cdhit_result = aa.filter_seq(
                    df_seq=cdhit_input, 
                    similarity_threshold=threshold
                )
                n_clusters = cdhit_result["cluster"].nunique()
                
                # Get representative sequences (indices align because both were reset)
                representative_mask = cdhit_result['is_representative'] == 1
                n_representatives = representative_mask.sum()
                
                print(f"  → {n_clusters} clusters, {n_representatives} representatives")
                
                # Filter original subset using representative mask
                subset_reset = subset_df
                filtered_subset = subset_reset[representative_mask].copy().reset_index(drop=True)
                filtered_subsets.append(filtered_subset)
                
            except Exception as e:
                print(f"  → Error in CD-HIT clustering: {str(e)}, keeping all sequences")
                filtered_subsets.append(subset_df)
        
        # Concatenate all filtered subsets
        if filtered_subsets:
            result_df = pd.concat(filtered_subsets, ignore_index=True)
        else:
            result_df = pd.DataFrame()
        
        return result_df
    
    def _segement_negative_windows(self, windows_df: pd.DataFrame, negative_ratio: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Segment negative windows into multiple balanced sub groups
        
        Args:
            windows_df: DataFrame containing all windows (positive and negative)
            negative_ratio: Ratio of negative to positive samples per segment
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing n balanced datasets
                Keys: 'dataset_1', 'dataset_2', ..., 'dataset_n'
                Values: DataFrames with balanced positive and negative samples
        """
        # Separate positive and negative windows
        positive_windows = windows_df[windows_df['known_cleavage_site'] == 1].copy()
        negative_windows = windows_df[windows_df['known_cleavage_site'] == 0].copy()
        
        n_positive = len(positive_windows)
        n_negative = len(negative_windows)
        
        print(f"\nSegmenting negative sample datasets:")
        print(f"  Number of positive samples: {n_positive}")
        print(f"  Number of negative samples: {n_negative}")
        print(f"  Negative to positive ratio per dataset: {negative_ratio}")
        
        if n_positive == 0:
            print("  Warning: No positive samples, cannot create balanced datasets")
            return {}
        
        if n_negative == 0:
            print("  Warning: No negative samples, cannot create balanced datasets")
            return {'dataset_1': positive_windows}
        
        # Calculate number of negative samples per segment
        neg_samples_per_segment = int(n_positive * negative_ratio)
        
        if neg_samples_per_segment == 0:
            print("  Warning: Number of negative samples per segment is 0, set to 1")
            neg_samples_per_segment = 1
        
        # Calculate number of segments needed
        n_segments = int(np.ceil(n_negative / neg_samples_per_segment))
        
        print(f"  Negative samples per segment: {neg_samples_per_segment}")
        print(f"  Expected number of datasets to create: {n_segments}")
        
        # Shuffle negative windows to ensure random distribution
        negative_windows_shuffled = negative_windows.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create balanced datasets
        balanced_datasets = {}
        
        for i in range(n_segments):
            # Calculate start and end indices for this segment
            start_idx = i * neg_samples_per_segment
            end_idx = min((i + 1) * neg_samples_per_segment, n_negative)
            
            # Get negative samples for this segment
            segment_negatives = negative_windows_shuffled.iloc[start_idx:end_idx].copy()
            
            # Combine with all positive samples
            balanced_dataset = pd.concat([positive_windows, segment_negatives], ignore_index=True)
            
            # Shuffle the combined dataset
            balanced_dataset = balanced_dataset.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            
            dataset_name = f'dataset_{i+1}'
            balanced_datasets[dataset_name] = balanced_dataset
            
            print(f"  {dataset_name}: {len(positive_windows)} positive samples + {len(segment_negatives)} negative samples = {len(balanced_dataset)} total samples")
        
        print(f"  Successfully created {len(balanced_datasets)} balanced datasets")
        return balanced_datasets


    def generate_windows(self, 
                        df: pd.DataFrame,
                        uniprot_id_col: str = 'final_entry',
                        cleavage_sites_col: str = 'final_cleavage_site',
                        extracellular_regions_col: str = 'extracellular',
                        transmembrane_regions_col: str = 'transmembrane',
                        intracellular_regions_col: str = 'intracellular',
                        window_type: str = "combined",
                        min_distance_from_cleavage: int = 10,
                        enable_distance_filter: bool = True,
                        max_sequence_distance: int = 150,
                        max_spatial_distance: float = 100.0,
                        structure_source: str = "alphafold",
                        padding_char: str = 'X',
                        cd_hit_clustering: bool = False,
                        threshold: float = 0.4,
                        show_topology: bool = True,
                        segement_negative_windows: bool = False,
                        segement_ratio: float = 1.0
                        ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate sliding windows for proteins with flexible options
        
        Args:
            df: Dataframe containing protein information
            window_type: Type of windows to generate ("positive", "negative", "combined")
                - "positive": only cleavage site windows
                - "negative": only negative windows (non-cleavage sites)  
                - "combined": both positive and negative windows
            min_distance_from_cleavage: Minimum distance from cleavage sites (in amino acids) for negative windows
            enable_distance_filter: Whether to enable distance filtering
            max_sequence_distance: Maximum sequence distance (amino acids)
            max_spatial_distance: Maximum spatial distance (Å)
            structure_source: Structure source for spatial analysis
            padding_char: Padding character for windows
            cd_hit_clustering: Whether to perform CD-HIT clustering
            threshold: Similarity threshold for CD-HIT clustering
            show_topology: Whether to include topology information in output
            segement_negative_windows: Whether to segment negative windows into balanced subsets
            segement_ratio: Ratio of negative to positive samples per segment
            
        Returns:
            Union[pd.DataFrame, Dict[str, pd.DataFrame]]: 
                If segement_negative_windows=False: Single DataFrame containing window information
                If segement_negative_windows=True: Dictionary containing multiple balanced datasets
        """
        # Validate window_type parameter
        valid_types = ["positive", "negative", "combined"]
        if window_type not in valid_types:
            raise ValueError(f"window_type must be one of {valid_types}, got '{window_type}'")
        
        all_windows = []
        
        print(f"Starting window generation with parameters:")
        print(f"  Window sizes: {self.window_sizes}")
        print(f"  Window type: {window_type}")
        print(f"  Distance filtering: {enable_distance_filter}")
        if enable_distance_filter:
            print(f"  Max sequence distance: {max_sequence_distance}")
            print(f"  Max spatial distance: {max_spatial_distance}")
            print(f"  Structure source: {structure_source}")
        print(f"  Min distance from cleavage: {min_distance_from_cleavage}")
        print(f"  Segment negative windows: {segement_negative_windows}")
        if segement_negative_windows:
            print(f"  Segment ratio: {segement_ratio}")
        
        for _, row in df.iterrows():
            uniprot_id = row[uniprot_id_col]
            cleavage_sites = row[cleavage_sites_col]
            extracellular_regions = row[extracellular_regions_col]
            transmembrane_regions = row[transmembrane_regions_col]
            
            # Check for intracellular regions (may not exist in all datasets)
            intracellular_regions = row.get(intracellular_regions_col, None) if intracellular_regions_col in row else None
            
            print(f"\nProcessing protein: {uniprot_id}")
            
            try:
                # get the protein sequence
                sequence = self._get_sequence_for_uniprot(uniprot_id)
                sequence_length = len(sequence)
                
                # Unified process: Get all qualified extracellular sites first
                if enable_distance_filter:
                    # Use distance filtering to get all qualified extracellular sites
                    all_qualified_sites = self._filter_residues_by_distance(
                        uniprot_id=uniprot_id,
                        extracellular_regions=extracellular_regions,
                        transmembrane_regions=transmembrane_regions,
                        intracellular_regions=intracellular_regions,
                        sequence_length=sequence_length,
                        enable_distance_filter=enable_distance_filter,
                        max_sequence_distance=max_sequence_distance,
                        max_spatial_distance=max_spatial_distance,
                        structure_source=structure_source
                    )
                else:
                    # Get all extracellular sites without distance filtering
                    if extracellular_regions:
                        all_qualified_sites = []
                        for start, end in extracellular_regions:
                            min_pos, max_pos = min(start, end), max(start, end)
                            all_qualified_sites.extend(range(min_pos, max_pos + 1))
                    else:
                        # Use distance_analyzer's method to get extracellular regions
                        inferred_extracellular_regions = self.distance_analyzer.get_extracellular_regions(
                            sequence_length=sequence_length,
                            intracellular_regions=intracellular_regions,
                            transmembrane_regions=transmembrane_regions
                        )
                        
                        # Convert regions to sites list
                        all_qualified_sites = []
                        for start, end in inferred_extracellular_regions:
                            min_pos, max_pos = min(start, end), max(start, end)
                            all_qualified_sites.extend(range(min_pos, max_pos + 1))

                # Now select target sites based on window type
                if window_type == "positive":
                    # Only cleavage sites that are in qualified extracellular sites
                    target_sites = [site for site in cleavage_sites if site in all_qualified_sites]
                    print(f"  Generating positive windows for {len(target_sites)} cleavage sites")
                    
                elif window_type == "negative":
                    # All qualified sites except cleavage sites
                    candidate_negative_sites = [site for site in all_qualified_sites if site not in cleavage_sites]
                    
                    # Apply minimum distance filtering for negative windows
                    if min_distance_from_cleavage > 0:
                        filtered_negative_sites = []
                        total_filtered = 0
                        print(f"  Filtering negative sites by minimum distance ({min_distance_from_cleavage} AA) from cleavage sites")
                        
                        for neg_site in candidate_negative_sites:
                            is_far_enough = True
                            for cleavage_pos in cleavage_sites:
                                distance = abs(neg_site - cleavage_pos)
                                if distance < min_distance_from_cleavage:
                                    is_far_enough = False
                                    break
                            
                            if is_far_enough:
                                filtered_negative_sites.append(neg_site)
                            else:
                                total_filtered += 1
                        
                        target_sites = filtered_negative_sites
                        print(f"  Filtered out {total_filtered} sites too close to cleavage sites")
                        print(f"  Generating negative windows for {len(target_sites)} sites")
                    else:
                        target_sites = candidate_negative_sites
                        print(f"  Generating negative windows for {len(target_sites)} sites")
                    
                elif window_type == "combined":
                    # All qualified sites (both positive and negative)
                    candidate_negative_sites = [site for site in all_qualified_sites if site not in cleavage_sites]
                    
                    # Apply minimum distance filtering for negative windows only
                    if min_distance_from_cleavage > 0:
                        filtered_negative_sites = []
                        total_filtered = 0
                        print(f"  Filtering negative sites by minimum distance ({min_distance_from_cleavage} AA) from cleavage sites")
                        
                        for neg_site in candidate_negative_sites:
                            is_far_enough = True
                            for cleavage_pos in cleavage_sites:
                                distance = abs(neg_site - cleavage_pos)
                                if distance < min_distance_from_cleavage:
                                    is_far_enough = False
                                    break
                            
                            if is_far_enough:
                                filtered_negative_sites.append(neg_site)
                            else:
                                total_filtered += 1
                        
                        # Combine positive sites with filtered negative sites
                        positive_sites_in_qualified = [site for site in cleavage_sites if site in all_qualified_sites]
                        target_sites = positive_sites_in_qualified + filtered_negative_sites
                        print(f"  Filtered out {total_filtered} negative sites too close to cleavage sites")
                        print(f"  Generating combined windows for {len(positive_sites_in_qualified)} positive + {len(filtered_negative_sites)} negative = {len(target_sites)} total sites")
                    else:
                        target_sites = all_qualified_sites
                        positive_count = len([site for site in target_sites if site in cleavage_sites])
                        negative_count = len(target_sites) - positive_count
                        print(f"  Generating combined windows for {positive_count} positive + {negative_count} negative = {len(target_sites)} total sites")
                
                # Generate windows for target sites
                for site in target_sites:
                    # Check if the residue is a known cleavage site
                    known_cleavage_site = 1 if site in cleavage_sites else 0
                    
                    # Generate windows of different sizes
                    for window_size in self.window_sizes:
                        window_seq = self._create_window(sequence, site, window_size, padding_char)
                        
                        window_info = {
                            'entry': uniprot_id,
                            'center_position': site, # 1-based index
                            'window_size': window_size,
                            'sequence': window_seq,
                            'center_residue': sequence[site-1] if site <= len(sequence) else padding_char,
                            'known_cleavage_site': known_cleavage_site,
                            'sequence_length': len(sequence)
                        }
                        
                        # Add topology information if requested
                        if show_topology:
                            window_info['extracellular'] = extracellular_regions
                            window_info['transmembrane'] = transmembrane_regions
                            window_info['intracellular'] = intracellular_regions
                        
                        all_windows.append(window_info)
                        
            except Exception as e:
                print(f"Error processing protein {uniprot_id}: {str(e)}")
                continue
        
        # Convert to DataFrame
        windows_df = pd.DataFrame(all_windows)
        
        if windows_df.empty:
            print("Warning: No windows generated")
            return {} if segement_negative_windows else windows_df
        
        
        # Apply CD-HIT clustering if requested
        if cd_hit_clustering:
            windows_df = self._cd_hit_clustering(windows_df, threshold, entry_col='entry', sequence_col='sequence')
        
        # Segment negative windows if requested
        if segement_negative_windows:
            # Only segment if we have both positive and negative samples
            if window_type == "combined":
                return self._segement_negative_windows(windows_df, segement_ratio)
            else:
                print("Warning: Segmentation of negative samples is only available when window_type='combined'")
                print("Returning a single dataset instead of a dictionary of segmented datasets")
                return windows_df
        
        return windows_df
    
    def generate_query_windows(self, 
                              df: pd.DataFrame,
                              uniprot_id_col: str = 'final_entry',
                              extracellular_regions_col: str = 'extracellular',
                              transmembrane_regions_col: str = 'transmembrane',
                              intracellular_regions_col: str = 'intracellular',
                              enable_distance_filter: bool = False,
                              max_sequence_distance: int = 150,
                              max_spatial_distance: float = 100.0,
                              structure_source: str = "alphafold",
                              padding_char: str = 'X',
                              show_topology: bool = True,
                              step_size: int = 1) -> pd.DataFrame:
        """
        Generate query windows for prediction (no cleavage site labels).
        Creates sliding windows across extracellular regions for proteins without known cleavage sites.
        
        Args:
            df: DataFrame with columns ['final_entry', 'extracellular', 'transmembrane', 'intracellular']
            uniprot_id_col: Column name for UniProt ID
            extracellular_regions_col: Column name for extracellular regions
            transmembrane_regions_col: Column name for transmembrane regions
            intracellular_regions_col: Column name for intracellular regions
            enable_distance_filter: Whether to enable distance filtering
            max_sequence_distance: Maximum sequence distance (amino acids)
            max_spatial_distance: Maximum spatial distance (Å)
            structure_source: Structure source for spatial analysis
            padding_char: Padding character for windows
            show_topology: Whether to include topology information in output
            step_size: Step size for sliding window (default: 1, every position)
            
        Returns:
            pd.DataFrame: DataFrame containing query windows for prediction
                Columns: entry, center_position, window_size, sequence, center_residue, 
                        sequence_length, [topology columns if requested]
        """
        print(f"Generating query windows for prediction:")
        print(f"  Window sizes: {self.window_sizes}")
        print(f"  Step size: {step_size}")
        print(f"  Distance filtering: {enable_distance_filter}")
        if enable_distance_filter:
            print(f"  Max sequence distance: {max_sequence_distance}")
            print(f"  Max spatial distance: {max_spatial_distance}")
            print(f"  Structure source: {structure_source}")
        
        all_query_windows = []
        
        for _, row in df.iterrows():
            uniprot_id = row[uniprot_id_col]
            extracellular_regions = row[extracellular_regions_col]
            transmembrane_regions = row[transmembrane_regions_col]
            
            # Handle intracellular regions (may not exist in all datasets)
            intracellular_regions = row.get(intracellular_regions_col, None) if intracellular_regions_col in row else None
            
            print(f"\nProcessing query protein: {uniprot_id}")
            
            try:
                # Get the protein sequence
                sequence = self._get_sequence_for_uniprot(uniprot_id)
                sequence_length = len(sequence)
                
                print(f"  Sequence length: {sequence_length} residues")
                
                # Get all qualified extracellular sites
                if enable_distance_filter:
                    # Use distance filtering to get qualified extracellular sites
                    qualified_sites = self._filter_residues_by_distance(
                        uniprot_id=uniprot_id,
                        extracellular_regions=extracellular_regions,
                        transmembrane_regions=transmembrane_regions,
                        intracellular_regions=intracellular_regions,
                        sequence_length=sequence_length,
                        enable_distance_filter=enable_distance_filter,
                        max_sequence_distance=max_sequence_distance,
                        max_spatial_distance=max_spatial_distance,
                        structure_source=structure_source
                    )
                else:
                    # Get all extracellular sites without distance filtering
                    if extracellular_regions:
                        qualified_sites = []
                        for start, end in extracellular_regions:
                            min_pos, max_pos = min(start, end), max(start, end)
                            # Use step_size for sliding window
                            qualified_sites.extend(range(min_pos, max_pos + 1, step_size))
                        print(f"  Using provided extracellular regions: {len(qualified_sites)} sites")
                    else:
                        # Use distance_analyzer's method to get extracellular regions
                        inferred_extracellular_regions = self.distance_analyzer.get_extracellular_regions(
                            sequence_length=sequence_length,
                            intracellular_regions=intracellular_regions,
                            transmembrane_regions=transmembrane_regions
                        )
                        
                        # Convert regions to sites list with step_size
                        qualified_sites = []
                        for start, end in inferred_extracellular_regions:
                            min_pos, max_pos = min(start, end), max(start, end)
                            qualified_sites.extend(range(min_pos, max_pos + 1, step_size))
                        
                        print(f"  Inferred extracellular regions: {len(qualified_sites)} sites")
                
                if not qualified_sites:
                    print(f"  No qualified sites found for {uniprot_id}")
                    continue
                
                print(f"  Generating windows for {len(qualified_sites)} positions")
                
                # Generate query windows for all qualified sites
                for site in qualified_sites:
                    # Generate windows of different sizes
                    for window_size in self.window_sizes:
                        window_seq = self._create_window(sequence, site, window_size, padding_char)
                        
                        query_window_info = {
                            'entry': uniprot_id,
                            'center_position': site,  # 1-based index
                            'window_size': window_size,
                            'sequence': window_seq,
                            'center_residue': sequence[site-1] if site <= len(sequence) else padding_char,
                            'sequence_length': len(sequence)
                        }
                        
                        # Add topology information if requested
                        if show_topology:
                            query_window_info['extracellular'] = extracellular_regions
                            query_window_info['transmembrane'] = transmembrane_regions
                            query_window_info['intracellular'] = intracellular_regions
                        
                        all_query_windows.append(query_window_info)
                        
            except Exception as e:
                print(f"Error processing query protein {uniprot_id}: {str(e)}")
                continue
        
        # Convert to DataFrame
        query_windows_df = pd.DataFrame(all_query_windows)
        
        if query_windows_df.empty:
            print("Warning: No query windows generated")
            return query_windows_df
        
        print(f"\nQuery window generation completed:")
        print(f"  Total proteins processed: {len(df)}")
        print(f"  Total query windows generated: {len(query_windows_df)}")
        print(f"  Window size distribution:")
        for window_size in self.window_sizes:
            count = len(query_windows_df[query_windows_df['window_size'] == window_size])
            print(f"    Size {window_size}: {count} windows")
        
        return query_windows_df
