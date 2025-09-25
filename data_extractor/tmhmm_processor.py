import re
import pandas as pd
from typing import Dict, List, Any, Optional

class TMHMMParser:
    """Parse the TMHMM topology prediction results"""
    
    @staticmethod
    def parse_tmhmm_file(tmhmm_text: str) -> List[Dict[str, Any]]:
        """
        Parse the TMHMM output file
        
        Args:
            tmhmm_text (str): the content of the TMHMM output file
            
        Returns:
            List[Dict[str, Any]]: a list of protein topology information
        """
        # Split the file by protein entries
        proteins = {}
        current_protein_id = None
        
        for line in tmhmm_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Comment lines with metadata
            if line.startswith('#'):
                # Extract protein ID and length from comment lines
                matches = re.search(r'# (sp|tr)\|([A-Z0-9\-]+)\|.*Length: (\d+)', line)
                if matches:
                    prefix, protein_id, length = matches.groups()
                    current_protein_id = f"{prefix}|{protein_id}"
                    
                    if current_protein_id not in proteins:
                        proteins[current_protein_id] = {
                            'uniprot_id': protein_id,
                            'prefix': prefix,
                            'length': int(length),
                            'regions': [],
                            'metadata': [],
                            'n_in_prob': None
                        }
                
                # Extract Total prob of N-in
                n_in_match = re.search(r'Total prob of N-in:\s+(\d+\.\d+)', line)
                if n_in_match and current_protein_id:
                    proteins[current_protein_id]['n_in_prob'] = float(n_in_match.group(1))
                        
                # Store metadata
                if current_protein_id:
                    proteins[current_protein_id]['metadata'].append(line)
            
            # Data lines with topology information
            else:
                parts = line.split()  # Split by whitespace, not just tabs
                if len(parts) >= 5:
                    # Extract protein ID from data lines
                    id_parts = parts[0].split('|')
                    if len(id_parts) >= 3:
                        prefix, protein_id = id_parts[0], id_parts[1]

                        # Create protein entry if it doesn't exist
                        if current_protein_id not in proteins:
                            proteins[current_protein_id] = {
                                'uniprot_id': protein_id,
                                'prefix': prefix,
                                'length': 0,  # Will be updated later if available
                                'regions': [],
                                'metadata': [],
                                'n_in_prob': None
                            }
                        
                        # Parse topology region
                        region_type = parts[2]
                        start_pos = int(parts[3])
                        end_pos = int(parts[4])
                        
                        proteins[current_protein_id]['regions'].append({
                            'type': region_type,
                            'start': start_pos,
                            'end': end_pos
                        })
        
        # Convert to list and standardize region names
        result = []
        for protein_id, data in proteins.items():
            # Skip proteins with no topology regions
            if not data['regions']:
                continue
                
            # Create standardized topology regions
            topology_regions = {
                "Signal_peptide": [],
                "Extracellular": [],
                "Transmembrane": [],
                "Cytoplasmic": []
            }
            
            for region in data['regions']:
                region_range = f"{region['start']}-{region['end']}"
                if region['type'] == 'outside':
                    topology_regions["Extracellular"].append(region_range)
                elif region['type'] == 'TMhelix':
                    topology_regions["Transmembrane"].append(region_range)
                elif region['type'] == 'inside':
                    topology_regions["Cytoplasmic"].append(region_range)
            
            # Build topology string (similar to TOPCONS format)
            if data['length'] > 0:
                topology_string = ['X'] * data['length']  # Initialize with placeholder
                for region in data['regions']:
                    for i in range(region['start']-1, region['end']):
                        if i < len(topology_string):
                            if region['type'] == 'outside':
                                topology_string[i] = 'o'
                            elif region['type'] == 'TMhelix':
                                topology_string[i] = 'M'
                            elif region['type'] == 'inside':
                                topology_string[i] = 'i'
            else:
                topology_string = []
            
            result.append({
                'uniprot_id': data['uniprot_id'],
                'length': data['length'],
                'topology_regions': topology_regions,
                'topology_string': ''.join(topology_string),
                'metadata': '\n'.join(data['metadata']),
                'n_in_prob': data['n_in_prob']
            })
            
        return result
    
    @staticmethod
    def generate_topology_dataframe(topology_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate the pandas DataFrame containing the protein topology information
        
        Args:
            topology_data (List[Dict[str, Any]]): the topology data returned by parse_tmhmm_file
            
        Returns:
            pd.DataFrame: the pandas DataFrame containing the protein topology information
        """
        # If no data, return empty DataFrame with columns
        if not topology_data:
            return pd.DataFrame(columns=[
                'uniprot_id', 'length', 'n_in_prob',
                'signal_peptide', 'extracellular', 'transmembrane', 'cytoplasmic'
            ])
            
        data = []
        for protein in topology_data:
            regions = protein['topology_regions']
            
            # Handle the case with multiple regions (convert the list to a comma-separated string)
            extra = ', '.join(regions['Extracellular']) if regions['Extracellular'] else None
            trans = ', '.join(regions['Transmembrane']) if regions['Transmembrane'] else None
            cyto = ', '.join(regions['Cytoplasmic']) if regions['Cytoplasmic'] else None
            
            data.append({
                'uniprot_id': protein['uniprot_id'],
                'length': protein['length'],
                'n_in_prob': protein['n_in_prob'],
                'extracellular': extra,
                'transmembrane': trans,
                'cytoplasmic': cyto
            })
        
        return pd.DataFrame(data)
        
