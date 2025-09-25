import pandas as pd
import numpy as np
from collections import defaultdict

from .structure_downloader import StructureDownloader
from .structure_paser import StructureParser
from .cache_manager import CacheManager


class DistanceAnalysis(CacheManager): 
    def __init__(self, cache_dir: str = "structure_temp/distance_cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = True):
        """Initialize Distance Analysis with caching system."""
        # Initialize cache system (default only use RAM cache)
        super().__init__(cache_dir=cache_dir, 
                        use_disk_cache=use_disk_cache, 
                        use_ram_cache=use_ram_cache)
        
        # initialize distance analysis components
        self.downloader = StructureDownloader()
        self.parser = StructureParser()

    def print_cache(self):
        """print the cache"""
        print("Distance Analysis Cache:")
        self.print_cache_stats()
        print("\nStructure Downloader Cache:")
        self.downloader.print_cache_stats()


    def _vertical_distance_analysis(self, uniprot_id, structure_source, 
                                    sites):
        """Vertical distance analysis (distance to membrane)"""
        
        _, filtered_pdb = self.downloader.read_pdb_by_source(structure_source=structure_source, 
                                                   uniprot_id=uniprot_id,pdb_id=None,chain=None)
                
        # Get membrane z-coordinates
        membrane_z_coords = []
        for line in filtered_pdb.split('\n'):
            if line.startswith('HETATM') and 'DUM' in line:
                z = float(line[46:54])
                membrane_z_coords.append(z)

        if not membrane_z_coords:
            raise ValueError(f"No membrane z-coordinates found in membranome for {uniprot_id}")

        upper_membrane = max(membrane_z_coords)

        site_distances = []
        for site in sites:
            # Check if site coordinates can be found
            if not self.parser._get_residue_coords(filtered_pdb, site):
                print(f"Warning: Site {site} can not be found in this file")
                continue

            site_coords = self.parser._get_residue_coords(filtered_pdb, site)
            if site_coords:
                distance = site_coords['z'] - upper_membrane
                site_distances.append({'site': site,
                                    'distance_type': 'Vertical',
                                    'distance': distance,
                                    'unit': 'Å'})
                
        return pd.DataFrame(site_distances)
    
    
    def _calculate_min_membrane_sequence_distance(self, site, transmembrane_regions):
        """Calculate the minimum distance to the nearest transmembrane region"""
        
        closest_points = []
        for tm_start, tm_end in transmembrane_regions:
            start_dist = abs(site - tm_start)
            end_dist = abs(site - tm_end)
            if start_dist < end_dist:
                closest_points.append( (tm_start, start_dist) )
            else:
                closest_points.append( (tm_end, end_dist) )
        
        membrane_site, min_dist = min(closest_points, key=lambda x: x[1])
        
        return membrane_site, min_dist


    def _sequence_distance_analysis(self, sites, transmembrane_regions):
        """Sequence distance analysis between residues and membrane"""
        # Calculate sequence distance and membrane distance between residues
        results = []
        for site in sites:
            # Calculate the shortest distance to the membrane
            membrane_site, min_membrane_dist = self._calculate_min_membrane_sequence_distance(
                    site, transmembrane_regions=transmembrane_regions
            )
            if min_membrane_dist is None:
                continue
            else:
                results.append({
                'site': site,
                'distance_type': 'Sequence',
                'membrane_site': membrane_site,
                'distance': min_membrane_dist,
                'unit': 'residues(aa)'
                })
        
        return pd.DataFrame(results)


    def _coordination_distance_analysis(self, uniprot_id, pdb_id=None, chain=None, sites=None, 
                                      structure_source="alphafold", 
                                      transmembrane_regions=None):
        """
        Spatial coordination distance analysis (distance to transmembrane region)
        The minimum distance from the site to the transmembran boundary on each possible chain will be calculated
        """

        valid_chains, filtered_pdb = self.downloader.read_pdb_by_source(structure_source=structure_source, 
                                                   uniprot_id=uniprot_id,pdb_id=pdb_id,chain=chain)

        site_chain_map = defaultdict(list) 

        for site in sites:
            for c in valid_chains:
                if self.parser._get_residue_coords(filtered_pdb, site, chain_id=c):
                    site_chain_map[site].append(c)
                else:
                    print(f"Warning: Site {site} not found in chain {c} in the PDB file")   

        # Build transmembrane region boundaries with chain information
        tm_boundaries = []
        for tm in transmembrane_regions:
            for c in valid_chains:
                # Collect all transmembrane region sites that exist in the PDB file
                tm_sites = []
                for pos in [tm[0], tm[1]]:
                    if self.parser._get_residue_coords(filtered_pdb, pos, chain_id=c):
                        tm_sites.append((c, pos))
                if tm_sites:  # Only add when valid sites are found
                    tm_boundaries.extend(tm_sites)
        if not tm_boundaries:
            if pdb_id:
                raise ValueError(f"No transmembrane regions found for {pdb_id}")
            else:
                raise ValueError(f"No transmembrane regions found for {uniprot_id}")

        distances = []
        for site in sites:
            if not self.parser._get_residue_coords(filtered_pdb, site):
                print(f"Warning: Site {site} not found in the PDB file")
                continue
            
            current_chains = site_chain_map[site] # None if site not found in any chain
            if not current_chains:
                continue

            for possible_chain in current_chains:
                # Only process the current chain's transmembrane regions
                same_chain_tm = [(c, pos) for c, pos in tm_boundaries if c == possible_chain]
                if not same_chain_tm:
                    print(f"Warning: No transmembrane regions found for site {site} in chain {possible_chain}")
                    continue
                
                # Get the coordinates of the site in the current chain
                site_coord = self.parser._get_residue_coords(filtered_pdb, site, chain_id=possible_chain)
                
                # Calculate the minimum distance in the current chain
                distance_list = []
                for _, pos in same_chain_tm:
                    distance = np.linalg.norm([
                            site_coord['x'] - self.parser._get_residue_coords(filtered_pdb, pos, chain_id=possible_chain)['x'],
                            site_coord['y'] - self.parser._get_residue_coords(filtered_pdb, pos, chain_id=possible_chain)['y'],
                            site_coord['z'] - self.parser._get_residue_coords(filtered_pdb, pos, chain_id=possible_chain)['z']
                        ])
                    distance_list.append(distance)
                chain_min = min(distance_list)
                membrane_site = same_chain_tm[distance_list.index(chain_min)][1]

                distances.append({
                        'site': site,
                        'chain': possible_chain,
                        'membrane_site': membrane_site,
                        'distance_type': 'Coordination',
                        'distance': round(chain_min, 2),
                        'unit': 'Å'
                        })

        return pd.DataFrame(distances)
    

    def get_extracellular_regions(self, sequence_length=None, 
                                  intracellular_regions=None, 
                                  transmembrane_regions=None):
        """Get extracellular regions from intracellular and transmembrane regions, in case of no extracellular regions provided"""
        # Create set of all excluded positions (transmembrane + intracellular)
        excluded_positions = set()
        # Add transmembrane positions
        for start, end in transmembrane_regions:
            min_pos, max_pos = min(start, end), max(start, end)
            excluded_positions.update(range(min_pos, max_pos + 1))
                    
        # Add intracellular positions
        if intracellular_regions:
            for start, end in intracellular_regions:
                min_pos, max_pos = min(start, end), max(start, end)
                excluded_positions.update(range(min_pos, max_pos + 1))
                    
        # Infer extracellular positions as all positions not in excluded set
        all_positions = set(range(1, sequence_length + 1))
        all_extracellular_sites = sorted(list(all_positions - excluded_positions))
                    
        # Convert back to regions for consistency with distance_analyzer
        extracellular_regions = []
        if all_extracellular_sites:
            # Group consecutive positions into regions
            start = all_extracellular_sites[0]
            end = start
                        
            for pos in all_extracellular_sites[1:]:
                if pos == end + 1:
                    end = pos
                else:
                    extracellular_regions.append((start, end))
                    start = pos
                    end = pos
            extracellular_regions.append((start, end))  # Add the last region
        
        return extracellular_regions


    def distance_analysis(self, sites, uniprot_id=None, pdb_id=None, chain= None, distance_type="vertical", 
                         extracellular_regions=None, transmembrane_regions=None,
                         structure_source="alphafold", 
                         only_extracellular_sites=True):
        """
        Distance analysis method
        Parameters:
            uniprot_id: UniProt ID, required
            pdb_id: PDB ID, required when using PDB structure
            chain: Chain identifier, required when using PDB structure, "A","B","AB", etc.
            distance_type: Distance type ["vertical", "sequence", "coordination"]
            vertical: Vertical distance between membrane and residue
            sequence: Sequence distance between residues 
            coordination: Coordination distance between atoms
            extracellular_regions: List of extracellular regions [(start1, end1), ...]
            transmembrane_regions: List of transmembrane regions [(start1, end1), ...]
            structure_source: Structure source ["membranome", "alphafold", "pdb"]
            sites: Sites to analyze, e.g. [123, 456, 789]
            only_extracellular_sites: Only analyze sites in extracellular regions
        """
        if distance_type != "sequence" and not uniprot_id:
            raise ValueError("Must provide UniProt ID")
        if not sites:
            raise ValueError("Must provide at least one analysis site")
        
        if structure_source not in ["membranome", "alphafold", "pdb"]:
            raise ValueError("structure_source must be one of 'membranome', 'alphafold', or 'pdb'")
            
        if distance_type not in ["vertical", "sequence", "coordination"]:
            raise ValueError("distance_type must be one of 'vertical', 'sequence', or 'coordination'")
        
        if extracellular_regions is None:
            raise ValueError("Must provide extracellular regions")
        if transmembrane_regions is None:
            raise ValueError("Must provide transmembrane regions")
        
        if only_extracellular_sites:
            sites_in_extracellular = []
            for site in sites:
                site_in_extracellular = False
                for start, end in extracellular_regions:
                    if start <= site <= end or end <= site <= start:
                        site_in_extracellular = True
                        break
                if site_in_extracellular:
                    sites_in_extracellular.append(site)
            sites = sites_in_extracellular

        # Select processing method based on distance type
        if distance_type == "vertical":
            if structure_source != "membranome":
                raise ValueError("Vertical distance analysis must use membranome structure")
            
            return self._vertical_distance_analysis(
                uniprot_id=uniprot_id,
                structure_source=structure_source,
                transmembrane_regions=transmembrane_regions,
                sites=sites
            )
        elif distance_type == "sequence":
            return self._sequence_distance_analysis(
                sites=sites,
                transmembrane_regions=transmembrane_regions
            )
        elif distance_type == "coordination":
            if structure_source == "pdb" and pdb_id is None:
                raise ValueError("PDB ID is required for PDB source file")
            return self._coordination_distance_analysis(
                uniprot_id=uniprot_id,
                pdb_id=pdb_id, chain=chain, 
                sites=sites,
                transmembrane_regions=transmembrane_regions,
                structure_source=structure_source
            )
        else:
            raise ValueError("Only support distance type: {'vertical', 'sequence', 'coordination'}")