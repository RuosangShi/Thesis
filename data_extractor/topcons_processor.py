import re
import pandas as pd
from typing import Dict, List, Any

class TopconsParser:
    """Parse the TOPCONS topology prediction results"""
    
    @staticmethod
    def parse_topcons_file(topcons_text: str) -> List[Dict[str, Any]]:
        """
        parse the topcons output file
        
        Args:
            topcons_text (str): the content of the topcons output file
            
        Returns:
            List[Dict[str, Any]]: a list of protein topology information
        """
        # use the sequence delimiter to split the file
        # TOPCONS usually has "Sequence number: X" to mark the start of a new sequence
        sequences = []
        current_seq = {}
        current_section = None
        
        for line in topcons_text.split('\n'):
            # the start of a new sequence
            if line.startswith('Sequence number:'):
                if current_seq:  # if there is already a sequence data, save it
                    sequences.append(current_seq)
                current_seq = {'raw_lines': [line]}
                current_section = 'header'
                
            # the name of the sequence
            elif line.startswith('Sequence name:') and current_section == 'header':
                # extract the uniprot id - match any format like xx|UniprotID|, xx|UniprotID-N|
                matches = re.search(r'\|([A-Z0-9\-]+)\|', line)
                if matches:
                    current_seq['uniprot_id'] = matches.group(1)
                current_seq['fasta_sequence'] = line.replace('Sequence name:', '').strip()
                current_seq['raw_lines'].append(line)
                
            # the length of the sequence
            elif line.startswith('Sequence length:') and current_section == 'header':
                length_match = re.search(r'(\d+)', line)
                if length_match:
                    current_seq['length'] = int(length_match.group(1))
                current_seq['raw_lines'].append(line)
                
            # the content of the sequence
            elif line.startswith('Sequence:') and current_section == 'header':
                current_section = 'sequence'
                current_seq['raw_lines'].append(line)
                
            # the start of the topology prediction
            elif line.startswith('TOPCONS predicted topology:'):
                current_section = 'topology'
                current_seq['raw_lines'].append(line)
                
            # the line of the topology prediction result
            elif current_section == 'topology' and line and not line.startswith('OCTOPUS'):
                if 'topology' not in current_seq:
                    current_seq['topology'] = line
                current_seq['raw_lines'].append(line)
                
            # OCTOPUS prediction (optional)
            elif line.startswith('OCTOPUS predicted topology:'):
                current_section = 'octopus'
                current_seq['raw_lines'].append(line)
                
            # other lines
            elif current_seq:
                current_seq['raw_lines'].append(line)
        
        # add the last sequence
        if current_seq:
            sequences.append(current_seq)
            
        # parse the topology information of each sequence
        parsed_results = []
        for seq in sequences:
            if 'topology' in seq:
                topology_regions = TopconsParser.parse_topology_string(seq['topology'])
                parsed_results.append({
                    'uniprot_id': seq.get('uniprot_id', 'unknown'),
                    'fasta_sequence': seq.get('fasta_sequence', ''),
                    'length': seq.get('length', 0),
                    'topology_string': seq.get('topology', ''),
                    'topology_regions': topology_regions,
                    'raw_data': '\n'.join(seq.get('raw_lines', []))
                })
                
        return parsed_results
    
    @staticmethod
    def parse_topology_string(topology_line: str) -> Dict[str, str]:
        """
        parse the topology string to get the structure regions
        
        Args:
            topology_line (str): the topology prediction string
            
        Returns:
            Dict[str, str]: a dictionary containing the structure regions
        """
        result = {
            "Signal_peptide": [],
            "Extracellular": [],
            "Transmembrane": [],
            "Cytoplasmic": []
        }
        
        # analyze the topology string
        current_state = None
        start_pos = 1  # 1-based indexing
        
        for i, char in enumerate(topology_line):
            if current_state is None:
                current_state = char
                start_pos = i + 1
            elif char != current_state:
                end_pos = i
                
                if current_state == 'S':
                    result["Signal_peptide"].append(f"{start_pos}-{end_pos}")
                elif current_state == 'o':
                    result["Extracellular"].append(f"{start_pos}-{end_pos}")
                elif current_state == 'M':
                    result["Transmembrane"].append(f"{start_pos}-{end_pos}")
                elif current_state == 'i':
                    result["Cytoplasmic"].append(f"{start_pos}-{end_pos}")
                
                current_state = char
                start_pos = end_pos + 1
        
        # handle the last region
        end_pos = len(topology_line)
        if current_state == 'S':
            result["Signal_peptide"].append(f"{start_pos}-{end_pos}")
        elif current_state == 'o':
            result["Extracellular"].append(f"{start_pos}-{end_pos}")
        elif current_state == 'M':
            result["Transmembrane"].append(f"{start_pos}-{end_pos}")
        elif current_state == 'i':
            result["Cytoplasmic"].append(f"{start_pos}-{end_pos}")
            
        return result
    

    @staticmethod
    def generate_topology_dataframe(topology_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        generate the pandas DataFrame containing the protein topology information
        
        Args:
            topology_data (List[Dict[str, Any]]): the topology data returned by parse_topcons_file
            
        Returns:
            pd.DataFrame: the pandas DataFrame containing the protein topology information
        """
        data = []
        for protein in topology_data:
            regions = protein['topology_regions']
            
            # handle the case with multiple regions (convert the list to a comma-separated string)
            signal = ', '.join(regions['Signal_peptide']) if regions['Signal_peptide'] else None
            extra = ', '.join(regions['Extracellular']) if regions['Extracellular'] else None
            trans = ', '.join(regions['Transmembrane']) if regions['Transmembrane'] else None
            cyto = ', '.join(regions['Cytoplasmic']) if regions['Cytoplasmic'] else None
            
            data.append({
                'uniprot_id': protein['uniprot_id'],
                'fasta_sequence': protein['fasta_sequence'],
                'length': protein['length'],
                'signal_peptide': signal,
                'extracellular': extra,
                'transmembrane': trans,
                'cytoplasmic': cyto
            })
        
        return pd.DataFrame(data)