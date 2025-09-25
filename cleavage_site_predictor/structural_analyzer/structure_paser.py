from Bio.PDB import DSSP, PDBParser


class StructureParser:
    def __init__(self):
        pass
    
    def _get_residue_coords(self, pdb_data, residue_num, chain_id=None):
        """obtain residue coordinates from pdb data"""
        if chain_id is None:
            for line in pdb_data.split('\n'):
                if line.startswith('ATOM'):
                    if int(line[22:26].strip()) == residue_num:
                        return {
                        'x': float(line[30:38]),
                        'y': float(line[38:46]),
                        'z': float(line[46:54])
                        }
        else:
            for line in pdb_data.split('\n'):
                if line.startswith('ATOM'):
                    if line[21].strip() == chain_id:
                        if int(line[22:26].strip()) == residue_num:
                            return {
                                'x': float(line[30:38]),
                                'y': float(line[38:46]),
                                'z': float(line[46:54])
                            }
        return None
    

    def _calculate_rsa(self, pdb_path, rsa_cal='Wilke'):
        """Calculate RSA with error handling."""
        from .structure_downloader import StructureFormatter

        adjusted_file = f"{pdb_path.rsplit('.', 1)[0]}_adjusted.pdb"
        print(pdb_path)
        print(adjusted_file)
        # Format the PDB file to ensure compliance
        StructureFormatter.format_pdb(pdb_path, adjusted_file)

        try:
            structure = PDBParser().get_structure('protein', adjusted_file)

            # Run DSSP on the structure
            dssp = DSSP(structure[0], adjusted_file, acc_array=rsa_cal)

            # Extract RSA values into a dictionary
            rsa_dict = {}
            for key in dssp.keys():
                residue_num = key[1][1] 
                rsa = dssp[key][3]       
                rsa_dict[residue_num] = rsa

            return rsa_dict

        except Exception as e:
            print(f"Warning: RSA calculation failed - {str(e)}")
            return {}
            
    
    def _get_chain_identifiers(self, pdb_data):
        """Get actual chain identifiers from PDB data
        Args:
            pdb_data: pdb data
        Returns:
            chains: set of chain identifiers
        """
        chains = set()
        for line in pdb_data.split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21].strip()  # Remove possible spaces
                if chain_id:  # Ignore empty chain identifiers
                    chains.add(chain_id)
        return chains
    

    def _filter_pdb_by_chain(self, pdb_data, chains):
        """filter pdb data by chain
        Args:
            pdb_data: pdb data
            chains: set/list of chain identifiers or single chain identifier
        Returns:
            filtered_pdb: filtered pdb data
        """

        filtered = []
        # Handle different input types for chains
        if isinstance(chains, str):
            specified_chains = {chains.upper()}
        elif isinstance(chains, (list, tuple, set)):
            specified_chains = {str(chain).upper() for chain in chains}
        else:
            specified_chains = {str(chains).upper()}
            
        for line in pdb_data.split('\n'):
            if line.startswith(('ATOM', 'TER')): 
                res_name = line[17:20].strip()
                #if res_name == 'HOH':
                    #continue
                # standard PDB format, the 22nd column (index 21) is the chain identifier
                chain_id = line[21].strip().upper() if len(line) > 21 else ''
                if chain_id in specified_chains:
                    filtered.append(line)
            elif line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name == 'DUM':  # keep DUM records for membrane boundary
                    filtered.append(line)
                else:
                    continue
            #elif line.startswith('TER'):
                # keep TER records to maintain structure integrity
                #filtered.append(line)
        return '\n'.join(filtered)
    
    
    def _save_filtered_pdb(self, filtered_pdb_content):
        """save filtered pdb content to temporary file and return file path"""
        
        import tempfile
        import os
        
        temp_dir = os.path.join(os.path.dirname(__file__), 'structure_temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # create temp file in temp folder
        tmp_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + '.pdb')
        with open(tmp_path, 'w') as tmp:
            tmp.write(filtered_pdb_content)
        
        return tmp_path