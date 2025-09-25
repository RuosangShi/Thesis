"""
Structure comparison module - responsible for comparing structural features between AlphaFold and PDB structures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .structural_profiler import (
    DistanceProfiler, ResidueDepthProfiler, DSSPProfiler, 
    FlexibilityProfiler
)
from ..window_slicer import WindowSlicer
from .structure_downloader import StructureDownloader


class StructureComparison:
    """
    Compare structural features between AlphaFold and PDB structures
    
    This class extracts various structural features for qualified residues in the extracellular domain
    and compares the results between AlphaFold and PDB structures.
    """
    
    def __init__(self, uniprot_timeout: int = 30):
        """
        Initialize the structure comparison tool
        
        Args:
            uniprot_timeout: Timeout for UniProt API requests (seconds)
        """
        self.uniprot_timeout = uniprot_timeout
        
        # Initialize component analyzers
        self.window_slicer = WindowSlicer(uniprot_timeout=uniprot_timeout)
        self.structure_downloader = StructureDownloader()
        self.distance_profiler = DistanceProfiler()
        self.depth_profiler = ResidueDepthProfiler()
        self.dssp_profiler = DSSPProfiler()
        self.flexibility_profiler = FlexibilityProfiler()
            
        print(f"StructureComparison initialized:")
        print(f"  UniProt timeout: {self.uniprot_timeout}s")
    
    def _get_qualified_residues(self, 
                              df: pd.DataFrame,
                              uniprot_id_col: str = 'final_entry',
                              cleavage_sites_col: str = 'final_cleavage_site',
                              extracellular_regions_col: str = 'extracellular',
                              transmembrane_regions_col: str = 'transmembrane',
                              intracellular_regions_col: str = 'intracellular',
                              enable_distance_filter: bool = True,
                              max_sequence_distance: int = 150,
                              max_spatial_distance: float = 100.0,
                              structure_source: str = "alphafold") -> pd.DataFrame:
        """
        Get all qualified residues in the extracellular domain using distance filtering
        
        Args:
            df: DataFrame containing protein information
            enable_distance_filter: Whether to enable distance filtering
            max_sequence_distance: Maximum sequence distance (amino acids)
            max_spatial_distance: Maximum spatial distance (Å)
            structure_source: Structure source for spatial analysis
            
        Returns:
            DataFrame with qualified residues for each protein
        """
        print(f"Getting qualified residues with parameters:")
        print(f"  Distance filtering: {enable_distance_filter}")
        if enable_distance_filter:
            print(f"  Max sequence distance: {max_sequence_distance}")
            print(f"  Max spatial distance: {max_spatial_distance}")
            print(f"  Structure source: {structure_source}")
        
        all_residues = []
        
        for _, row in df.iterrows():
            uniprot_id = row[uniprot_id_col]
            cleavage_sites = row[cleavage_sites_col]
            extracellular_regions = row[extracellular_regions_col]
            transmembrane_regions = row[transmembrane_regions_col]
            intracellular_regions = row.get(intracellular_regions_col, None) if intracellular_regions_col in row else None
            
            print(f"\nProcessing protein: {uniprot_id}")

            if not transmembrane_regions:
                print(f"Warning: Missing transmembrane information for {uniprot_id}")
                continue
            
            try:
                # Get the protein sequence and sequence length for THIS protein
                sequence = self.window_slicer._get_sequence_for_uniprot(uniprot_id)
                sequence_length = len(sequence)
                
                # Handle empty extracellular regions using window_slicer strategy
                if not extracellular_regions:
                    print(f"  No extracellular regions provided for {uniprot_id}, inferring...")
                        
                    # Use distance_analyzer's method to get extracellular regions
                    extracellular_regions = self.distance_profiler.distance_analyzer.get_extracellular_regions(
                        sequence_length=sequence_length,
                        transmembrane_regions=transmembrane_regions,
                        intracellular_regions=intracellular_regions
                    )
                        
                    print(f"  Inferred {len(extracellular_regions)} extracellular regions")
                    
                if not extracellular_regions:
                    print(f"Warning: No extracellular regions found for {uniprot_id}")
                    continue
                
                # Use distance filtering to get all qualified extracellular sites
                qualified_sites = self.window_slicer._filter_residues_by_distance(
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
                
                # Create records for each qualified residue
                for site in qualified_sites:
                    residue_info = {
                        'entry': uniprot_id,
                        'position': site,  # 1-based index
                        'residue': sequence[site-1] if site <= len(sequence) else 'X',
                        'is_cleavage_site': 1 if site in cleavage_sites else 0,
                        'sequence_length': sequence_length,
                        'extracellular': extracellular_regions,
                        'transmembrane': transmembrane_regions,
                        'intracellular': intracellular_regions
                    }
                    all_residues.append(residue_info)
                
                print(f"  Found {len(qualified_sites)} qualified residues")
                
            except Exception as e:
                print(f"Error processing protein {uniprot_id}: {str(e)}")
                continue
        
        residues_df = pd.DataFrame(all_residues)
        
        if not residues_df.empty:
            print(f"\nTotal qualified residues: {len(residues_df)}")
            print(f"Unique proteins: {residues_df['entry'].nunique()}")
            print(f"Cleavage sites: {residues_df['is_cleavage_site'].sum()}")
            print(f"Non-cleavage sites: {(residues_df['is_cleavage_site'] == 0).sum()}")
        else:
            print("No qualified residues found")
        
        return residues_df

    def _extract_structure_features(self, 
                                   uniprot_id: str,
                                   sites: list,
                                   structure_source: str = "alphafold",
                                   pdb_id: str = None,
                                   include_dssp: bool = True,
                                   include_depth: bool = True,
                                   include_distance: bool = True,
                                   extracellular_regions=None,
                                   transmembrane_regions=None) -> dict:
        """
        Analyze structural features for specified sites using various analyzers
        
        Args:
            uniprot_id: UniProt ID
            sites: List of sites to analyze (1-based)
            structure_source: Structure source ("alphafold" or "pdb")
            pdb_id: PDB ID (required if structure_source is "pdb")
            include_dssp: Include DSSP analysis (RSA, secondary structure, phi/psi angles)
            include_depth: Include residue depth analysis
            include_distance: Include distance analysis
            extracellular_regions, transmembrane_regions: Topology info for distance analysis
            
        Returns:
            Dict mapping site -> feature dict
        """
        if not sites:
            return {}
        
        print(f"      Analyzing {len(sites)} sites with {structure_source} structure...")
        
        # Initialize result dictionary
        site_features = {}
        for site in sites:
            site_features[site] = {}
        
        # DSSP Analysis (RSA, secondary structure)
        if include_dssp:
            try:
                dssp_results = self.dssp_profiler.dssp_analyzer.run_dssp_analysis(
                    uniprot_id=uniprot_id,
                    sites=sites,
                    pdb_id=pdb_id,
                    rsa_cal='Wilke',
                    structure_source=structure_source
                )
                
                if not dssp_results.empty:
                    # Handle column name - could be 'position' or 'site'
                    position_col = 'site' if 'site' in dssp_results.columns else 'position'
                    
                    for _, row in dssp_results.iterrows():
                        site = row[position_col]
                        if site in site_features:
                            site_features[site]['rsa'] = row.get('rsa', None)
                            site_features[site]['secondary_structure'] = row.get('dssp', None)
                            # Extract phi and psi angles from DSSP
                            site_features[site]['phi'] = row.get('phi', None)
                            site_features[site]['sin_phi'] = np.sin(site_features[site]['phi'] * np.pi / 180) if site_features[site]['phi'] is not None else None
                            site_features[site]['cos_phi'] = np.cos(site_features[site]['phi'] * np.pi / 180) if site_features[site]['phi'] is not None else None
                            site_features[site]['psi'] = row.get('psi', None)
                            site_features[site]['sin_psi'] = np.sin(site_features[site]['psi'] * np.pi / 180) if site_features[site]['psi'] is not None else None
                            site_features[site]['cos_psi'] = np.cos(site_features[site]['psi'] * np.pi / 180) if site_features[site]['psi'] is not None else None
                            
            except Exception as e:
                print(f"        Warning: DSSP analysis failed: {str(e)}")
        
        # Residue Depth Analysis
        if include_depth:
            try:
                depth_results = self.depth_profiler.residue_depth_analyzer.run_residue_depth_analysis(
                    uniprot_id=uniprot_id,
                    sites=sites,
                    pdb_id=pdb_id,
                    structure_source=structure_source
                )
                
                if not depth_results.empty:
                    # Handle column name - could be 'position' or 'site'
                    position_col = 'site' if 'site' in depth_results.columns else 'position'
                    
                    for _, row in depth_results.iterrows():
                        site = row[position_col]
                        if site in site_features:
                            site_features[site]['residue_depth'] = row.get('residue_depth', None)
                            site_features[site]['relative_depth'] = row.get('relative_depth', None)
                            
            except Exception as e:
                print(f"        Warning: Depth analysis failed: {str(e)}")
        
        # Distance Analysis (sequence and spatial distances to membrane)
        if include_distance and transmembrane_regions:
            try:
                # Sequence distance analysis
                seq_distance_results = self.distance_profiler.distance_analyzer.distance_analysis(
                    sites=sites,
                    uniprot_id=uniprot_id,
                    distance_type="sequence",
                    extracellular_regions=extracellular_regions,
                    transmembrane_regions=transmembrane_regions,
                    only_extracellular_sites=True
                )
                
                if not seq_distance_results.empty:
                    for _, row in seq_distance_results.iterrows():
                        site = row['site']
                        distance_value = row.get('distance', None)
                        if site in site_features and distance_value is not None:
                            site_features[site]['sequence_distance'] = distance_value
                
                # Spatial distance analysis (only if structure available)
                if structure_source in ["alphafold", "pdb"]:
                    spatial_distance_results = self.distance_profiler.distance_analyzer.distance_analysis(
                        sites=sites,
                        uniprot_id=uniprot_id,
                        pdb_id=pdb_id if structure_source == "pdb" else None,
                        distance_type="coordination",
                        extracellular_regions=extracellular_regions,
                        transmembrane_regions=transmembrane_regions,
                        structure_source=structure_source,
                        only_extracellular_sites=True
                    )
                    
                    if not spatial_distance_results.empty:
                        for _, row in spatial_distance_results.iterrows():
                            site = row['site']
                            distance_value = row.get('distance', None)
                            if site in site_features and distance_value is not None:
                                site_features[site]['spatial_distance'] = distance_value
                                
            except Exception as e:
                print(f"        Warning: Distance analysis failed: {str(e)}")
        
        
        # Count successful features
        successful_sites = []
        for site, features in site_features.items():
            if features and any(v is not None for v in features.values()):
                successful_sites.append(site)
        
        print(f"        Successfully extracted features for {len(successful_sites)}/{len(sites)} sites")
        
        return site_features

    def _extract_pdb_features(self, 
                                 uniprot_id: str,
                                 target_sites: list,
                                 include_dssp: bool = True,
                                 include_depth: bool = True,
                                 include_distance: bool = True,
                                 extracellular_regions=None,
                                 transmembrane_regions=None) -> tuple:
        """
        Download all PDB structures and aggregate features for sites with multiple structure results
        
        Args:
            uniprot_id: UniProt ID
            target_sites: List of target sites to analyze
            include_*: Boolean flags for different analysis types
            extracellular_regions, transmembrane_regions, intracellular_regions: Topology info
            
        Returns:
            Tuple of (pdb_features_dict, coverage_info)
            - pdb_features_dict: {site: {'features': {...}, 'structures_used': [pdb_ids], 'num_structures': int}}
            - coverage_info: {'structures_downloaded': [pdb_ids], 'total_structures': int}
        """
        # Get prioritized PDB list (metadata only)
        prioritized_pdb_list = self.structure_downloader.get_prioritized_pdb_list(uniprot_id)
        
        if not prioritized_pdb_list:
            print(f"    No PDB structures available for {uniprot_id}")
            return {}, {'structures_downloaded': [], 'total_structures': 0}
        
        print(f"    Found {len(prioritized_pdb_list)} PDB structures for {uniprot_id}")
        print(f"    Will download all structures: {[entry['pdb_id'] for entry in prioritized_pdb_list[:5]]}" + 
              (f" ... (+{len(prioritized_pdb_list)-5} more)" if len(prioritized_pdb_list) > 5 else ""))
        
        # Download all structures and collect features
        site_structure_features = {}  # {site: {pdb_id: features}}
        structures_downloaded = []
        structure_metadata = {}  # {pdb_id: {resolution, resolution_str}}
        
        # Step 1: Download all structures and extract features
        for entry in prioritized_pdb_list:
            pdb_id = entry['pdb_id']
            resolution = entry['resolution']
            resolution_str = entry['resolution_str']
            
            print(f"    Processing {pdb_id} (resolution: {resolution_str})...")
            
            try:
                # Download and analyze this structure
                structure_info = self.structure_downloader.download_and_analyze_pdb(uniprot_id, pdb_id)
                
                if not structure_info:
                    print(f"      ✗ Failed to download/process {pdb_id}, skipping")
                    continue
                
                # Find sites that this structure can cover
                covered_positions = set(structure_info['covered_positions'])
                structure_target_sites = list(set(target_sites) & covered_positions)
                
                if not structure_target_sites:
                    print(f"      ℹ {pdb_id}: No target sites available in this structure")
                    continue
                
                print(f"      Target sites for this structure: {len(structure_target_sites)}")
                
                # Analyze features for this structure
                structure_features = self._extract_structure_features(
                    uniprot_id=uniprot_id,
                    sites=structure_target_sites,
                    structure_source="pdb",
                    pdb_id=pdb_id,
                    include_dssp=include_dssp,
                    include_depth=include_depth,
                    include_distance=include_distance,
                    extracellular_regions=extracellular_regions,
                    transmembrane_regions=transmembrane_regions
                )
                
                # Collect features for each site
                successful_sites = 0
                for site in structure_target_sites:
                    if site in structure_features and structure_features[site]:
                        features = structure_features[site]
                        if any(v is not None for v in features.values()):
                            # Initialize site entry if needed
                            if site not in site_structure_features:
                                site_structure_features[site] = {}
                            
                            # Store features for this structure
                            site_structure_features[site][pdb_id] = features
                            successful_sites += 1
                
                print(f"      ✓ Successfully extracted features for {successful_sites}/{len(structure_target_sites)} sites")
                
                if successful_sites > 0:
                    structures_downloaded.append(pdb_id)
                    structure_metadata[pdb_id] = {
                        'resolution': resolution,
                        'resolution_str': resolution_str
                    }
                
            except Exception as e:
                print(f"      ✗ Error processing {pdb_id}: {str(e)}")
                continue
        
        # Step 2: Aggregate features for each site across multiple structures
        print(f"    Aggregating features across {len(structures_downloaded)} downloaded structures...")
        aggregated_features = self._aggregate_site_features(site_structure_features, structure_metadata)
        
        # Final summary
        total_target = len(target_sites)
        total_covered = len(aggregated_features)
        coverage_percentage = (total_covered / total_target * 100) if total_target > 0 else 0
        
        print(f"    Feature aggregation complete:")
        print(f"      Total target sites: {total_target}")
        print(f"      Sites with aggregated features: {total_covered} ({coverage_percentage:.1f}%)")
        print(f"      Structures downloaded: {len(structures_downloaded)}")
        
        # Print site coverage statistics
        if aggregated_features:
            structure_counts = [len(data['structures_used']) for data in aggregated_features.values()]
            avg_structures = sum(structure_counts) / len(structure_counts)
            max_structures = max(structure_counts)
            print(f"      Average structures per site: {avg_structures:.1f}")
            print(f"      Maximum structures per site: {max_structures}")
        
        coverage_info = {
            'structures_downloaded': structures_downloaded,
            'total_structures': len(structures_downloaded),
            'total_target': total_target,
            'total_covered': total_covered,
            'coverage_percentage': coverage_percentage
        }
        
        return aggregated_features, coverage_info 

    def _aggregate_site_features(self, site_structure_features: dict, structure_metadata: dict) -> dict:
        """
        Aggregate features from multiple structures for each site
        
        Args:
            site_structure_features: {site: {pdb_id: features}}
            structure_metadata: {pdb_id: {resolution, resolution_str}}
            
        Returns:
            Dict: {site: {'features': aggregated_features, 'structures_used': [pdb_ids], 'num_structures': int}}
        """
        from collections import Counter
        import numpy as np
        
        aggregated_features = {}
        
        # Define which features should be averaged vs take mode
        continuous_features = ['rsa', 'residue_depth', 'relative_depth', 'phi', 'psi', 
                             'sequence_distance', 'spatial_distance']
        categorical_features = ['secondary_structure']
        
        for site, structure_features in site_structure_features.items():
            if not structure_features:
                continue
                
            # Collect all feature values from different structures
            feature_collections = {}
            structures_used = list(structure_features.keys())
            
            # Initialize feature collections
            all_feature_names = set()
            for pdb_id, features in structure_features.items():
                all_feature_names.update(features.keys())
            
            for feature_name in all_feature_names:
                feature_collections[feature_name] = []
                
                # Collect values from all structures that have this feature
                for pdb_id, features in structure_features.items():
                    if feature_name in features and features[feature_name] is not None:
                        feature_collections[feature_name].append(features[feature_name])
            
            # Aggregate features
            aggregated_site_features = {}
            
            for feature_name, values in feature_collections.items():
                if not values:
                    aggregated_site_features[feature_name] = None
                    continue
                
                try:
                    if feature_name in continuous_features:
                        # Take average for continuous variables
                        numeric_values = []
                        for val in values:
                            try:
                                # Handle potential array/list values (convert to scalar)
                                if hasattr(val, '__len__') and not isinstance(val, str):
                                    if len(val) > 0:
                                        for item in val:
                                            if not pd.isna(item):
                                                numeric_values.append(float(item))
                                                break  # Take first valid value
                                    else:
                                        continue
                                else:
                                    numeric_values.append(float(val))
                            except (ValueError, TypeError):
                                continue
                        
                        if numeric_values:
                            aggregated_site_features[feature_name] = np.mean(numeric_values)
                        else:
                            aggregated_site_features[feature_name] = None
                            
                    elif feature_name in categorical_features:
                        # Take mode (most frequent) for categorical variables
                        str_values = [str(val) for val in values if val is not None and str(val) != 'nan']
                        if str_values:
                            counter = Counter(str_values)
                            most_common = counter.most_common(1)[0][0]
                            aggregated_site_features[feature_name] = most_common
                        else:
                            aggregated_site_features[feature_name] = None
                    else:
                        # For unknown features, try averaging first, then mode
                        try:
                            numeric_values = []
                            for val in values:
                                try:
                                    if hasattr(val, '__len__') and not isinstance(val, str):
                                        if len(val) > 0:
                                            for item in val:
                                                if not pd.isna(item):
                                                    numeric_values.append(float(item))
                                                    break
                                    else:
                                        numeric_values.append(float(val))
                                except (ValueError, TypeError):
                                    continue
                            
                            if numeric_values:
                                aggregated_site_features[feature_name] = np.mean(numeric_values)
                            else:
                                # Try categorical approach
                                str_values = [str(val) for val in values if val is not None and str(val) != 'nan']
                                if str_values:
                                    counter = Counter(str_values)
                                    most_common = counter.most_common(1)[0][0]
                                    aggregated_site_features[feature_name] = most_common
                                else:
                                    aggregated_site_features[feature_name] = None
                        except:
                            aggregated_site_features[feature_name] = None
                            
                except Exception as e:
                    print(f"        Warning: Error aggregating feature {feature_name} for site {site}: {str(e)}")
                    aggregated_site_features[feature_name] = None
            
            # Store aggregated results
            aggregated_features[site] = {
                'features': aggregated_site_features,
                'structures_used': structures_used,
                'num_structures': len(structures_used)
            }
        
        print(f"      Aggregated features for {len(aggregated_features)} sites")
        if aggregated_features:
            # Print aggregation statistics
            total_structures_per_site = [data['num_structures'] for data in aggregated_features.values()]
            avg_structures = sum(total_structures_per_site) / len(total_structures_per_site)
            
            # Count how many sites used multiple structures
            multi_structure_sites = sum(1 for count in total_structures_per_site if count > 1)
            single_structure_sites = len(total_structures_per_site) - multi_structure_sites
            
            print(f"      Sites with single structure: {single_structure_sites}")
            print(f"      Sites with multiple structures: {multi_structure_sites}")
            print(f"      Average structures per site: {avg_structures:.2f}")
            
            # Show examples of aggregation
            if multi_structure_sites > 0:
                example_site = None
                for site, data in aggregated_features.items():
                    if data['num_structures'] > 1:
                        example_site = site
                        break
                if example_site:
                    example_data = aggregated_features[example_site]
                    print(f"      Example multi-structure site {example_site}: used {example_data['structures_used']}")
        
        return aggregated_features

    def _clean_feature_data(self, df: pd.DataFrame, af_col: str, pdb_col: str) -> pd.DataFrame:
        """Clean feature data by converting sequences to scalars if needed"""
        def clean_value(val):
            if val is None or pd.isna(val):
                return val
            # Check if the value is a sequence (list, array, etc.) but not a string
            if hasattr(val, '__len__') and not isinstance(val, str):
                try:
                    # If it's a sequence, try to get the first non-NaN element
                    if len(val) > 0:
                        for item in val:
                            if not pd.isna(item):
                                return float(item)
                    return None
                except (TypeError, ValueError):
                    return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return val
        
        # Clean both columns
        df = df.copy()  # Make a copy to avoid modifying original
        df[af_col] = df[af_col].apply(clean_value)
        df[pdb_col] = df[pdb_col].apply(clean_value)
        
        return df

    def _print_feature_comparison_stats(self, df: pd.DataFrame, features: List[str], stats: Dict) -> None:
        """Print detailed statistics for feature comparisons"""
        print("\n" + "="*60)
        print("ALPHAFOLD VS PDB FEATURE COMPARISON STATISTICS")
        print("="*60)
        
        total_sites = len(df)
        print(f"Total sites analyzed: {total_sites}")
        
        if 'has_pdb_data' in df.columns:
            sites_with_pdb = df['has_pdb_data'].sum()
            print(f"Sites with any PDB data: {sites_with_pdb} ({100*sites_with_pdb/total_sites:.1f}%)")
        
        print(f"\nFeature-specific statistics:")
        
        for feature in features:
            af_col = f'af_{feature}'
            pdb_col = f'pdb_{feature}'
            
            print(f"\n{feature.upper()}:")
            
            # Data availability
            feature_stats = stats[feature]
            print(f"  AlphaFold data: {feature_stats['af_available']}/{total_sites} ({100*feature_stats['af_available']/total_sites:.1f}%)")
            print(f"  PDB data: {feature_stats['pdb_available']}/{total_sites} ({feature_stats['pdb_coverage']:.1f}%)")
            print(f"  Both available: {feature_stats['both_available']}/{total_sites} ({feature_stats['comparison_coverage']:.1f}%)")
            
            # Calculate correlations and differences for valid data
            valid_mask = df[af_col].notna() & df[pdb_col].notna()
            if valid_mask.sum() >= 3:
                valid_data = df[valid_mask]
                af_values = valid_data[af_col]
                pdb_values = valid_data[pdb_col]
                
                try:
                    # Handle secondary structure as categorical variable
                    if feature == 'secondary_structure':
                        # Calculate accuracy for categorical data
                        af_ss = af_values.tolist()
                        pdb_ss = pdb_values.tolist()
                        correct = sum(1 for af, pdb in zip(af_ss, pdb_ss) if af == pdb)
                        accuracy = correct / len(af_ss) * 100
                        
                        # Get unique types and their counts
                        all_ss_types = sorted(list(set(af_ss + pdb_ss)))
                        af_counts = {ss: af_ss.count(ss) for ss in all_ss_types}
                        pdb_counts = {ss: pdb_ss.count(ss) for ss in all_ss_types}
                        
                        print(f"  Overall accuracy: {accuracy:.2f}% ({correct}/{len(af_ss)})")
                        print(f"  Secondary structure types: {all_ss_types}")
                        print(f"  AlphaFold distribution: {af_counts}")
                        print(f"  PDB distribution: {pdb_counts}")
                        
                        # Per-class accuracy
                        for ss_type in all_ss_types:
                            ss_mask = [pdb == ss_type for pdb in pdb_ss]
                            if any(ss_mask):
                                ss_af = [af_ss[i] for i, mask in enumerate(ss_mask) if mask]
                                ss_pdb = [pdb_ss[i] for i, mask in enumerate(ss_mask) if mask]
                                ss_correct = sum(1 for af, pdb in zip(ss_af, ss_pdb) if af == pdb)
                                ss_accuracy = ss_correct / len(ss_af) * 100 if ss_af else 0
                                print(f"    {ss_type}: {ss_accuracy:.1f}% accuracy ({ss_correct}/{len(ss_af)})")
                    
                    else:
                        # Handle continuous variables
                        from scipy.stats import spearmanr
                        
                        # Correlations
                        pearson_r, pearson_p = pearsonr(af_values, pdb_values)
                        spearman_r, spearman_p = spearmanr(af_values, pdb_values)
                        
                        # Differences
                        differences = af_values - pdb_values
                        mae = np.mean(np.abs(differences))
                        rmse = np.sqrt(np.mean(differences ** 2))
                        
                        print(f"  Pearson correlation: r={pearson_r:.4f} (p={pearson_p:.2e})")
                        print(f"  Spearman correlation: ρ={spearman_r:.4f} (p={spearman_p:.2e})")
                        print(f"  Mean Absolute Error: {mae:.4f}")
                        print(f"  Root Mean Square Error: {rmse:.4f}")
                        print(f"  Mean difference (AF-PDB): {np.mean(differences):.4f} ± {np.std(differences):.4f}")
                        
                        # Range comparisons
                        print(f"  AlphaFold range: {af_values.min():.3f} - {af_values.max():.3f}")
                        print(f"  PDB range: {pdb_values.min():.3f} - {pdb_values.max():.3f}")
                    
                except Exception as e:
                    print(f"  Error calculating statistics: {str(e)}")
        
        print("="*60)

    def _clean_feature_data(self, df: pd.DataFrame, af_col: str, pdb_col: str) -> pd.DataFrame:
        """Clean feature data by converting sequences to scalars if needed"""
        def clean_value(val):
            if val is None or pd.isna(val):
                return val
            # Check if the value is a sequence (list, array, etc.) but not a string
            if hasattr(val, '__len__') and not isinstance(val, str):
                try:
                    # If it's a sequence, try to get the first non-NaN element
                    if len(val) > 0:
                        for item in val:
                            if not pd.isna(item):
                                return float(item)
                    return None
                except (TypeError, ValueError):
                    return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return val
        
        # Clean both columns
        df[af_col] = df[af_col].apply(clean_value)
        df[pdb_col] = df[pdb_col].apply(clean_value)
        
        return df

    def _plot_secondary_structure_comparison(self, ax, af_values, pdb_values, valid_data, feature_stats):

        """Plot secondary structure comparison using confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Convert to lists for easier handling
        af_ss = af_values.tolist()
        pdb_ss = pdb_values.tolist()
        
        # Get unique secondary structure types
        all_ss_types = sorted(list(set(af_ss + pdb_ss)))
        
        # Create confusion matrix
        cm = confusion_matrix(pdb_ss, af_ss, labels=all_ss_types)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot heatmap
        sns.heatmap(cm_percent, 
                   annot=True, 
                   fmt='.1f',
                   xticklabels=all_ss_types,
                   yticklabels=all_ss_types,
                   cmap='Blues',
                   ax=ax)
        
        ax.set_xlabel('AlphaFold Secondary Structure', fontweight='bold')
        ax.set_ylabel('PDB Secondary Structure', fontweight='bold')
        
        # Calculate overall accuracy
        correct = sum(1 for af, pdb in zip(af_ss, pdb_ss) if af == pdb)
        accuracy = correct / len(af_ss) * 100
        
        # Create title with statistics
        n_sites = len(valid_data)
        coverage = feature_stats['comparison_coverage']
        pdb_coverage = feature_stats['pdb_coverage']
        
        title = (f'Secondary Structure\n'
               f'Accuracy: {accuracy:.1f}%\n'
               f'n={n_sites} ({coverage:.1f}% of total sites)\n'
               f'PDB coverage: {pdb_coverage:.1f}%')
        
        ax.set_title(title, fontweight='bold', fontsize=10)
        
        # Print detailed secondary structure statistics
        print(f"\nSecondary Structure Comparison Details:")
        print(f"  Overall accuracy: {accuracy:.1f}% ({correct}/{len(af_ss)})")
        print(f"  Secondary structure types found: {all_ss_types}")
        
        # Count occurrences
        af_counts = {ss: af_ss.count(ss) for ss in all_ss_types}
        pdb_counts = {ss: pdb_ss.count(ss) for ss in all_ss_types}
        
        print(f"  AlphaFold distribution: {af_counts}")
        print(f"  PDB distribution: {pdb_counts}")
        
        # Per-class accuracy
        for ss_type in all_ss_types:
            ss_mask = [pdb == ss_type for pdb in pdb_ss]
            if any(ss_mask):
                ss_af = [af_ss[i] for i, mask in enumerate(ss_mask) if mask]
                ss_pdb = [pdb_ss[i] for i, mask in enumerate(ss_mask) if mask]
                ss_correct = sum(1 for af, pdb in zip(ss_af, ss_pdb) if af == pdb)
                ss_accuracy = ss_correct / len(ss_af) * 100 if ss_af else 0
                print(f"  {ss_type}: {ss_accuracy:.1f}% accuracy ({ss_correct}/{len(ss_af)})")
        
        return accuracy 
    
    def compare_alphafold_vs_pdb_aggregated(self, 
                          df: pd.DataFrame,
                          uniprot_id_col: str = 'final_entry',
                          cleavage_sites_col: str = 'final_cleavage_site',
                          extracellular_regions_col: str = 'extracellular',
                          transmembrane_regions_col: str = 'transmembrane',
                          intracellular_regions_col: str = 'intracellular',
                          enable_distance_filter: bool = True,
                          max_sequence_distance: int = 150,
                          max_spatial_distance: float = 100.0,
                          include_dssp: bool = True,
                          include_depth: bool = True,
                          include_distance: bool = True
                          ) -> pd.DataFrame:
        """
        Aggregated comparison between AlphaFold and PDB structures
        Downloads all available PDB structures and aggregates features for sites with multiple structure results:
        - Continuous variables (RSA, depth, angles): take average
        - Categorical variables (secondary structure): take mode (most frequent)
        - Conformation angles (phi, psi): extracted from DSSP only
        
        Args:
            df: DataFrame containing protein information
            enable_distance_filter: Whether to enable distance filtering
            max_sequence_distance: Maximum sequence distance (amino acids)
            max_spatial_distance: Maximum spatial distance (Å)
            include_dssp: Include DSSP analysis (RSA, secondary structure, phi/psi angles)
            include_depth: Include residue depth analysis
            include_distance: Include distance analysis
            
        Returns:
            DataFrame with aggregated comparison between AlphaFold and PDB features
        """
        print("=== Aggregated AlphaFold vs PDB Structure Comparison ===")
                # Step 1: Get all qualified residues using the unified method
        print("Getting qualified residues for comparison...")
        residues_df = self._get_qualified_residues(
            df=df,
            uniprot_id_col=uniprot_id_col,
            cleavage_sites_col=cleavage_sites_col,
            extracellular_regions_col=extracellular_regions_col,
            transmembrane_regions_col=transmembrane_regions_col,
            intracellular_regions_col=intracellular_regions_col,
            enable_distance_filter=enable_distance_filter,
            max_sequence_distance=max_sequence_distance,
            max_spatial_distance=max_spatial_distance,
            structure_source="alphafold"
        )
        
        if residues_df.empty:
            print("No qualified residues found for comparison")
            return pd.DataFrame()
        
        comparison_results = []
        
        # Process each unique protein
        for uniprot_id in residues_df['entry'].unique():
            protein_residues = residues_df[residues_df['entry'] == uniprot_id]
            if protein_residues.empty:
                continue
                
            print(f"\n=== Processing protein: {uniprot_id} ===")
            
            # Get protein info from first residue
            first_residue = protein_residues.iloc[0]
            cleavage_sites = df[df[uniprot_id_col] == uniprot_id][cleavage_sites_col].iloc[0]
            sequence_length = first_residue['sequence_length']
            extracellular_regions = first_residue['extracellular']
            transmembrane_regions = first_residue['transmembrane']
            intracellular_regions = first_residue['intracellular']
            
            # Get qualified sites for this protein
            qualified_sites = protein_residues['position'].tolist()
            print(f"  Target sites for comparison: {len(qualified_sites)}")
            
            try:
                # Get protein sequence
                sequence = self.window_slicer._get_sequence_for_uniprot(uniprot_id)
                
                # Analyze AlphaFold structure for all qualified sites
                print(f"  Analyzing AlphaFold structure...")
                alphafold_features = self._extract_structure_features(
                    uniprot_id=uniprot_id,
                    sites=qualified_sites,
                    structure_source="alphafold",
                    include_dssp=include_dssp,
                    include_depth=include_depth,
                    include_distance=include_distance,
                    extracellular_regions=extracellular_regions,
                    transmembrane_regions=transmembrane_regions
                )
                
                # Progressive PDB analysis
                print(f"  Starting progressive PDB analysis...")
                pdb_features, coverage_info = self._extract_pdb_features(
                    uniprot_id=uniprot_id,
                    target_sites=qualified_sites,
                    include_dssp=include_dssp,
                    include_depth=include_depth,
                    include_distance=include_distance,
                    extracellular_regions=extracellular_regions,
                    transmembrane_regions=transmembrane_regions
                )
                
                # Create comparison records
                for site in qualified_sites:
                    # Get AlphaFold features for this site
                    af_site_features = alphafold_features.get(site, {})
                    pdb_site_info = pdb_features.get(site, {})
                    pdb_site_features = pdb_site_info.get('features', {}) if pdb_site_info else {}
                    
                    comparison_record = {
                        'uniprot_id': uniprot_id,
                        'position': site,
                        'residue': sequence[site-1] if site <= len(sequence) else 'X',
                        'is_cleavage_site': 1 if site in cleavage_sites else 0,
                        'sequence_length': sequence_length,
                        'pdb_structures_used': pdb_site_info.get('structures_used', []) if pdb_site_info else [],
                        'pdb_num_structures': pdb_site_info.get('num_structures', 0) if pdb_site_info else 0,
                        'has_pdb_data': bool(pdb_site_info and pdb_site_features)
                    }
                    
                    # Add AlphaFold features with 'af_' prefix
                    for feature, value in af_site_features.items():
                        comparison_record[f'af_{feature}'] = value
                    
                    # Add aggregated PDB features with 'pdb_' prefix
                    for feature, value in pdb_site_features.items():
                        comparison_record[f'pdb_{feature}'] = value
                    
                    # Calculate differences where both values exist (only for continuous variables)
                    for feature in ['rsa', 'residue_depth', 'relative_depth', 'phi', 'psi']:
                        af_key = f'af_{feature}'
                        pdb_key = f'pdb_{feature}'
                        
                        if af_key in comparison_record and pdb_key in comparison_record:
                            af_val = comparison_record[af_key]
                            pdb_val = comparison_record[pdb_key]
                            
                            if af_val is not None and pdb_val is not None:
                                try:
                                    diff = float(af_val) - float(pdb_val)
                                    comparison_record[f'diff_{feature}'] = diff
                                    comparison_record[f'abs_diff_{feature}'] = abs(diff)
                                except (ValueError, TypeError):
                                    comparison_record[f'diff_{feature}'] = None
                                    comparison_record[f'abs_diff_{feature}'] = None
                    
                    comparison_results.append(comparison_record)
                
                # Print coverage summary
                total_sites = len(qualified_sites)
                covered_sites = len([site for site in qualified_sites if site in pdb_features])
                print(f"  Aggregated PDB coverage: {covered_sites}/{total_sites} sites ({100*covered_sites/total_sites:.1f}%)")
                print(f"  Structures downloaded: {coverage_info['structures_downloaded']}")
                print(f"  Total structures used: {coverage_info['total_structures']}")
                
            except Exception as e:
                print(f"Error processing protein {uniprot_id}: {str(e)}")
                continue
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            print(f"\n=== Aggregated Comparison Summary ===")
            print(f"Total comparison records: {len(comparison_df)}")
            print(f"Unique proteins: {comparison_df['uniprot_id'].nunique()}")
            print(f"Cleavage sites: {comparison_df['is_cleavage_site'].sum()}")
            print(f"Non-cleavage sites: {(comparison_df['is_cleavage_site'] == 0).sum()}")
            
            # PDB coverage statistics
            with_pdb = comparison_df['has_pdb_data'].sum()
            without_pdb = len(comparison_df) - with_pdb
            print(f"Sites with PDB data: {with_pdb}/{len(comparison_df)} ({100*with_pdb/len(comparison_df):.1f}%)")
            print(f"Sites without PDB data: {without_pdb}/{len(comparison_df)} ({100*without_pdb/len(comparison_df):.1f}%)")
            
            # Structure usage statistics
            if 'pdb_structures_used' in comparison_df.columns:
                # Flatten all structure lists and count usage
                all_structures = []
                for structures_list in comparison_df['pdb_structures_used']:
                    if isinstance(structures_list, list):
                        all_structures.extend(structures_list)
                
                if all_structures:
                    from collections import Counter
                    structure_counts = Counter(all_structures)
                    print(f"PDB structure usage (total uses across all sites):")
                    for structure, count in structure_counts.most_common(10):
                        print(f"  {structure}: {count} sites")
                    
                    # Multi-structure site statistics
                    multi_structure_sites = comparison_df[comparison_df['pdb_num_structures'] > 1]
                    if not multi_structure_sites.empty:
                        print(f"Sites using multiple structures: {len(multi_structure_sites)}")
                        avg_structures = comparison_df[comparison_df['pdb_num_structures'] > 0]['pdb_num_structures'].mean()
                        max_structures = comparison_df['pdb_num_structures'].max()
                        print(f"Average structures per covered site: {avg_structures:.2f}")
                        print(f"Maximum structures per site: {max_structures}")
            
            # Print data completeness for key features
            for feature in ['rsa', 'residue_depth', 'secondary_structure']:
                af_col = f'af_{feature}'
                pdb_col = f'pdb_{feature}'
                if af_col in comparison_df.columns and pdb_col in comparison_df.columns:
                    af_available = comparison_df[af_col].notna().sum()
                    pdb_available = comparison_df[pdb_col].notna().sum()
                    both_available = (comparison_df[af_col].notna() & comparison_df[pdb_col].notna()).sum()
                    print(f"  {feature}: AF={af_available}, PDB={pdb_available}, Both={both_available}")
            
        else:
            print("No comparison data generated")
        
        return comparison_df 

    def plot_alphafold_vs_pdb_features(self, 
                                  comparison_df: pd.DataFrame,
                                  features_to_plot: List[str] = None,
                                  figsize: Tuple[int, int] = (18, 12),
                                  color_by: str = "protein", 
                                  show_coverage_info: bool = True) -> None:
        """
        Plot AlphaFold vs PDB feature comparisons with flexible coloring options
        
        Args:
            comparison_df: DataFrame from compare_alphafold_vs_pdb_aggregated
            features_to_plot: List of features to plot
            figsize: Figure size
            color_by: Coloring strategy - "protein", "secondary_structure_af", or "secondary_structure_pdb"
            show_coverage_info: Whether to show coverage information
        """
        if comparison_df.empty:
            print("No comparison data available for plotting")
            return
        
        # Default features to plot
        if features_to_plot is None:
            features_to_plot = ['rsa', 'residue_depth', 'relative_depth', 'secondary_structure', 
                                'phi', 'psi', 'sin_phi', 'cos_phi', 'sin_psi', 'cos_psi']
        
        # Filter features that actually exist in the data
        available_features = []
        feature_stats = {}
        
        for feature in features_to_plot:
            af_col = f'af_{feature}'
            pdb_col = f'pdb_{feature}'
            
            if af_col in comparison_df.columns and pdb_col in comparison_df.columns:
                # Clean data - convert sequences to scalars if needed
                comparison_df = self._clean_feature_data(comparison_df, af_col, pdb_col)
                
                # Calculate statistics including NaN handling
                total_sites = len(comparison_df)
                af_available = comparison_df[af_col].notna().sum()
                pdb_available = comparison_df[pdb_col].notna().sum()
                both_available = (comparison_df[af_col].notna() & comparison_df[pdb_col].notna()).sum()
                
                if both_available >= 3:  # Need at least 3 points for meaningful comparison
                    available_features.append(feature)
                    feature_stats[feature] = {
                        'total_sites': total_sites,
                        'af_available': af_available,
                        'pdb_available': pdb_available,
                        'both_available': both_available,
                        'pdb_coverage': (pdb_available / total_sites) * 100,
                        'comparison_coverage': (both_available / total_sites) * 100
                    }
        
        if not available_features:
            print("No valid features available for plotting")
            return
        
        # Calculate subplot dimensions
        n_features = len(available_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Setup coloring scheme based on color_by parameter
        if color_by == "protein" and 'uniprot_id' in comparison_df.columns:
            unique_values = comparison_df['uniprot_id'].unique()
            color_col = 'uniprot_id'
            color_label = "Protein"
        elif color_by == "secondary_structure_af" and 'af_secondary_structure' in comparison_df.columns:
            unique_values = comparison_df['af_secondary_structure'].dropna().unique()
            color_col = 'af_secondary_structure'
            color_label = "AlphaFold Secondary Structure"
        elif color_by == "secondary_structure_pdb" and 'pdb_secondary_structure' in comparison_df.columns:
            unique_values = comparison_df['pdb_secondary_structure'].dropna().unique()
            color_col = 'pdb_secondary_structure'
            color_label = "PDB Secondary Structure"
        else:
            # Fallback to protein if specified column doesn't exist
            unique_values = comparison_df['uniprot_id'].unique() if 'uniprot_id' in comparison_df.columns else ['default']
            color_col = 'uniprot_id' if 'uniprot_id' in comparison_df.columns else None
            color_label = "Protein"
            print(f"Warning: Could not color by '{color_by}', falling back to protein coloring")
        
        # Create color mapping
        if len(unique_values) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))
        
        color_map = dict(zip(unique_values, colors))
        
        # Define secondary structure markers and colors for better visualization
        ss_markers = {
            'H': 'o',      # Alpha helix - circle
            'E': 's',      # Beta sheet - square  
            'B': '^',      # Beta bridge - triangle up
            'G': 'v',      # 3-10 helix - triangle down
            'I': '<',      # Pi helix - triangle left
            'T': '>',      # Turn - triangle right
            'S': 'D',      # Bend - diamond
            '-': '.',      # Coil/loop - point
            'C': 'P',      # Coil - plus
            'L': 'X'       # Loop - x
        }
        
        ss_colors = {
            'H': '#FF6B6B',    # Red for alpha helix
            'E': '#4ECDC4',    # Teal for beta sheet
            'B': '#45B7D1',    # Blue for beta bridge
            'G': '#96CEB4',    # Green for 3-10 helix
            'I': '#FFEAA7',    # Yellow for pi helix
            'T': '#DDA0DD',    # Plum for turn
            'S': '#98D8C8',    # Mint for bend
            '-': '#AED6F1',    # Light blue for coil
            'C': '#F7DC6F',    # Light yellow for coil
            'L': '#D2B4DE'     # Light purple for loop
        }
        
        # Plot each feature
        for idx, feature in enumerate(available_features):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            af_col = f'af_{feature}'
            pdb_col = f'pdb_{feature}'
            
            # Get data with both values available (non-NaN)
            valid_mask = comparison_df[af_col].notna() & comparison_df[pdb_col].notna()
            valid_data = comparison_df[valid_mask]
            
            if len(valid_data) == 0:
                ax.text(0.5, 0.5, f'No valid data for {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature}')
                continue
            
            alphafold_values = valid_data[af_col]
            pdb_values = valid_data[pdb_col]
            
            # Handle secondary structure as categorical variable
            if feature == 'secondary_structure':
                self._plot_secondary_structure_comparison(ax, alphafold_values, pdb_values, valid_data, feature_stats[feature])
                continue
            
            # Create scatter plot for continuous variables
            if color_col and color_col in valid_data.columns:
                # Group by color category
                for value in unique_values:
                    if pd.isna(value):
                        continue
                        
                    value_mask = valid_data[color_col] == value
                    if value_mask.sum() == 0:
                        continue
                    
                    value_data = valid_data[value_mask]
                    af_vals = alphafold_values[value_mask]
                    pdb_vals = pdb_values[value_mask]
                    
                    # Choose marker and color based on coloring scheme
                    if color_by.startswith("secondary_structure"):
                        marker = ss_markers.get(str(value), 'o')
                        color = ss_colors.get(str(value), color_map.get(value, 'gray'))
                    else:
                        marker = 'o'
                        color = color_map.get(value, 'gray')
                    
                    # Separate cleavage sites if available
                    if 'is_cleavage_site' in value_data.columns:
                        # Non-cleavage sites
                        non_cleavage = value_data['is_cleavage_site'] == 0
                        if non_cleavage.sum() > 0:
                            ax.scatter(af_vals[non_cleavage], pdb_vals[non_cleavage],
                                     c=[color], alpha=0.6, s=40, marker=marker,
                                     label=f'{value} (non-cleavage)' if idx == 0 else "",
                                     edgecolors='black', linewidths=0.5)
                        
                        # Cleavage sites
                        cleavage = value_data['is_cleavage_site'] == 1
                        if cleavage.sum() > 0:
                            ax.scatter(af_vals[cleavage], pdb_vals[cleavage],
                                     c=[color], alpha=0.9, s=80, marker=marker,
                                     edgecolors='red', linewidth=2,
                                     label=f'{value} (cleavage)' if idx == 0 else "")
                    else:
                        ax.scatter(af_vals, pdb_vals, c=[color], 
                                 marker=marker, alpha=0.7, s=50,
                                 label=f'{value}' if idx == 0 else "",
                                 edgecolors='black', linewidths=0.5)
            else:
                # Fallback: single color, distinguish cleavage sites
                if 'is_cleavage_site' in valid_data.columns:
                    non_cleavage = valid_data['is_cleavage_site'] == 0
                    cleavage = valid_data['is_cleavage_site'] == 1
                    
                    if non_cleavage.sum() > 0:
                        ax.scatter(alphafold_values[non_cleavage], pdb_values[non_cleavage], 
                                 c='lightblue', alpha=0.6, s=30, label='Non-cleavage sites')
                    if cleavage.sum() > 0:
                        ax.scatter(alphafold_values[cleavage], pdb_values[cleavage], 
                                 c='red', alpha=0.9, s=60, marker='^', 
                                 edgecolors='black', linewidth=1, label='Cleavage sites')
                else:
                    ax.scatter(alphafold_values, pdb_values, alpha=0.7, s=30, c='steelblue')
            
            # Add diagonal line (perfect correlation)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect correlation')
            
            # Add best fit line
            if len(alphafold_values) > 1:
                try:
                    z = np.polyfit(alphafold_values, pdb_values, 1)
                    p = np.poly1d(z)
                    ax.plot(alphafold_values, p(alphafold_values), "r-", alpha=0.8, linewidth=2, label='Best fit')
                except:
                    pass
            
            # Calculate and display statistics
            try:
                pearson_r, pearson_p = pearsonr(alphafold_values, pdb_values)
                mae = np.mean(np.abs(alphafold_values - pdb_values))
                
                # Create title with statistics and coverage info
                stats = feature_stats[feature]
                if show_coverage_info:
                    title = (f'{feature}\n'
                           f'r={pearson_r:.3f}, MAE={mae:.3f}\n'
                           f'n={len(valid_data)} ({stats["comparison_coverage"]:.1f}% of total sites)\n'
                           f'PDB coverage: {stats["pdb_coverage"]:.1f}%\n'
                           f'Colored by: {color_label}')
                else:
                    title = f'{feature}\nr={pearson_r:.3f}, MAE={mae:.3f}, n={len(valid_data)}'
                
            except Exception as e:
                title = f'{feature}\nn={len(valid_data)}'
            
            # Formatting
            ax.set_xlabel(f'AlphaFold {feature}', fontweight='bold')
            ax.set_ylabel(f'PDB {feature}', fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add legend for first subplot
            if idx == 0:
                legend_elements = ax.get_legend_handles_labels()
                if legend_elements[0]:  # If there are legend elements
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
        # Remove empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].remove()
            else:
                axes[col].remove()
        
        plt.suptitle(f'AlphaFold vs PDB Feature Comparison\nColored by: {color_label}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        self._print_feature_comparison_stats(comparison_df, available_features, feature_stats)
