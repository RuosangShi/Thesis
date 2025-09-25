import os
import re
import requests
import pandas as pd
from io import StringIO
from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from .structure_paser import StructureParser
from .cache_manager import CacheManager
from typing import List, Dict


class StructureFormatter:
    def __init__(self):
        self.parser = StructureParser()

    @staticmethod
    def format_pdb_content(pdb_content):
        """ process the pdb content
        Args:
            pdb_content (str): PDB File Content in string
        Returns:
            str: Formatted PDB Content
        """
        formatted_lines = []
        for line in pdb_content.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                record_type = line[:6].strip()
                serial = int(line[6:11].strip())
                atom_name = line[12:16].strip().ljust(4)
                alt_loc = line[16:17].strip() or " "
                res_name = line[17:20].strip().rjust(3)
                chain_id = line[21:22].strip() or "A"
                res_seq = int(line[22:26].strip())
                insert_code = line[26:27].strip() or " "
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip() or 1.00)
                temp_factor = float(line[60:66].strip() or 0.00)
                element = atom_name[0].rjust(2)
                formatted_line = (
                    f"{record_type:<6}{serial:5d} {atom_name:<4}{alt_loc}{res_name:>3} {chain_id}"
                    f"{res_seq:>4}{insert_code}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}"
                )
                formatted_lines.append(formatted_line)
            elif line.startswith("TER"):
                formatted_lines.append(line[:6].ljust(6) + line[6:11].strip().rjust(5))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    @staticmethod
    def format_pdb(input_path, output_path):
        """
        Format the PDB file.
        ______________
        Args:
            input_path: Path to the original PDB file
            output_path: Path to the formatted PDB file
        ______________
        Returns:
            str: The path to the formatted PDB file
        """
        with open(input_path,"r") as fin, open(output_path, 'w') as fout:
            for line in fin:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    record_type = line[:6].strip()
                    serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip().ljust(4)
                    alt_loc = line[16:17].strip() or " "
                    res_name = line[17:20].strip().rjust(3)
                    chain_id = line[21:22].strip() or "A"
                    res_seq = int(line[22:26].strip())
                    insert_code = line[26:27].strip() or " "
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    occupancy = float(line[54:60].strip() or 1.00)
                    temp_factor = float(line[60:66].strip() or 0.00)
                    element = atom_name[0].rjust(2)
                    formatted_line = (
                        f"{record_type:<6}{serial:5d} {atom_name:<4}{alt_loc}{res_name:>3} {chain_id}"
                        f"{res_seq:>4}{insert_code}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}\n"
                    )
                    fout.write(formatted_line)
                elif line.startswith("TER"):
                    fout.write(line[:6].ljust(6) + line[6:11].strip().rjust(5) + "\n")
                else:
                    fout.write(line)
        return output_path

    @staticmethod
    def process_fasta_sequence(fasta_str):
        """Process FASTA string to get sequence."""
        fasta = StringIO(fasta_str)
        for record in SeqIO.parse(fasta, "fasta"):
            return str(record.seq)
        return None

    @staticmethod
    def _get_uniprot_fasta(uniprot_id):
        """
        Retrieve full FASTA sequence by UniProt ID
        ______________
        Args:
            uniprot_id (str): Protein ID in UniProt database
        ______________
        Returns:
            str: Complete FASTA sequence string
        """

        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Failed to fetch FASTA sequence for UniProt ID {uniprot_id}, HTTP status: {response.status_code}")

    @staticmethod
    def fix_pdb_numbering(pdb_path, uniprot_id):
        """
        Align the residue numbering of the PDB file to the full UniProt sequence
        ______________
        Args:
            pdb_path: Path to the original PDB file
            uniprot_id: UniProt ID of the protein
        ______________
        Returns:
            str: The path to the fixed PDB file
        """
        full_str = StructureFormatter._get_uniprot_fasta(uniprot_id)
        full_seq = StructureFormatter.process_fasta_sequence(full_str)

        base_name = os.path.basename(pdb_path)
        fixed_path = os.path.join('pdb_structures', f"fixed_{base_name}")

        # parse the PDB structure
        parser = PDBParser()
        structure = parser.get_structure('temp', pdb_path)
        
        # process the sequence by chain
        chain_data = {}
        for chain in structure.get_chains():
            chain_id = chain.id
            pdb_residues = []
            res_numbers = []
            
            # extract the chain sequence and original numbering
            for res in chain:
                res_id = res.id # the residue id
                res_num = f"{res_id[1]}{res_id[2].strip()}"  # process the insertion code
                pdb_residues.append(res.resname) # the residue name
                res_numbers.append(res_num) # the residue number
            
            # convert to single letter sequence
            pdb_seq = "".join([seq1(r) for r in pdb_residues])
            
            # set the alignment parameters
            aligner = PairwiseAligner()
            aligner.mode = 'local'
            aligner.match = 2.0
            aligner.mismatch = -1.0
            aligner.open_gap_score = -0.5
            aligner.extend_gap_score = -0.1
            
            try:
                # execute the sequence alignment
                alignments = aligner.align(full_seq, pdb_seq)
                
                if alignments:
                    best_aln = alignments[0]
                    #alignment_score = best_aln.score
                    alignment_length = len(str(best_aln).split('\n')[0])
                    identity = sum(a == b for a, b in zip(str(best_aln).split('\n')[0], 
                                                    str(best_aln).split('\n')[2]))
                    identity_percentage = (identity / alignment_length) * 100

                    # if the alignment quality is too low, skip the chain
                    if identity_percentage < 30: 
                        print(f"Chain {chain_id} alignment identity too low ({identity_percentage:.1f}%), skipping...")
                        continue

                    # get the alignment interval correctly
                    target_blocks, query_blocks = best_aln.aligned
                    
                    # get the starting position of the first matching block (for local alignment)
                    aln_start_target = target_blocks[0][0]  # the starting position of the target sequence
                    aln_start_query = query_blocks[0][0]    # the starting position of the query sequence
                    
                    # generate the residue mapping table
                    res_map = {}
                    full_pos = aln_start_target  # the position of the full sequence
                    pdb_pos = aln_start_query    # the position of the PDB sequence
                    
                    # loop through the aligned sequences
                    for t_char, q_char in zip(best_aln[0], best_aln[1]):
                        if t_char != '-':
                            full_pos += 1
                        if q_char != '-':
                            pdb_pos += 1
                        if t_char != '-' and q_char != '-':
                            res_map[pdb_pos - 1] = full_pos  # convert the PDB index to 0-based
                    
                    chain_data[chain_id] = {
                        'res_numbers': res_numbers,
                        'res_map': res_map,
                        'aln_start': aln_start_target,
                        'aln_end': aln_start_target + len(best_aln[0]) - 1
                    }
                else:
                    print(f"No alignment found for chain {chain_id}, skipping...")
                    continue

            except Exception as e:
                print(f"Error processing chain {chain_id}: {str(e)}")
                continue

        if not chain_data:
            print("No chains could be aligned with the UniProt sequence. Returning original file.")
            return pdb_path

        # rewrite the PDB file
        with open(pdb_path) as fin, open(fixed_path, 'w') as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    chain_id = line[21]
                    if chain_id not in chain_data:
                        # skip the chain that cannot be aligned
                        continue
                    
                    orig_res = line[22:26].strip()
                    insert_code = line[26]  # keep the insertion code
                    
                    if chain_id in chain_data:
                        mapping = chain_data[chain_id]
                        try:
                            res_idx = mapping['res_numbers'].index(orig_res) # 0-based query index
                            full_num = mapping['res_map'].get(res_idx, None) # 1-based target index 
                            
                            if full_num is not None:
                                # format to the PDB standard column format
                                #new_res = f"{full_num + 1:4d}"  # convert to 1-based numbering
                                new_res = f"{full_num:4d}"
                                new_line = f"{line[:22]}{new_res}{insert_code}{line[27:]}"
                                fout.write(new_line)
                                continue
                            #else:
                                #new_res = "9999"
                                #new_line = f"{line[:22]}{new_res}{insert_code}{line[27:]}"
                                #fout.write(f"REMARK residue {orig_res} is not aligned\n")
                                #fout.write(new_line) 
                        except ValueError:
                            #fout.write(line)
                            continue
                    else:
                        #fout.write(line)
                        continue
                else:
                    fout.write(line)
        
        return fixed_path


class StructureDownloader(CacheManager):
    def __init__(self, cache_dir: str = "structure_temp/download_cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = False):
        """Initialize Structure Downloader with caching system."""
        # Parent class to initialize cache system (default only use RAM cache)
        super().__init__(cache_dir=cache_dir, 
                        use_disk_cache=use_disk_cache, 
                        use_ram_cache=use_ram_cache)
        
        # initialize structure download components
        self.parser = StructureParser()
        self.formatter = StructureFormatter()

    def print_cache(self):
        """print the cache"""
        self.print_cache_stats()

    def _get_structure_cache_key(self, uniprot_id=None, pdb_id=None, structure_source='alphafold'):
        """generate structure download cache key"""
        return self._generate_cache_key(
            uniprot_id=uniprot_id,
            pdb_id=pdb_id,
            structure_source=structure_source
        )

    def _check_file_exists(self, file_path):
        """check if the file exists and is not empty"""
        return file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def download_pdb_structure(self, uniprot_id=None, pdb_id=None, structure_source='alphafold'):
        """
        First checks pdb database, then alphafold db
        ______________
        Args:
            uniprot_id: Uniprot identifier of the protein
            pdb_id: PDB identifier of the protein
            structure_source: Download from alphafold db (alphafold)/ membranome (membranome)/ PDB databank (PDB)
        ______________
        Returns:
            str: Path to downloaded structure file or None if no structure found
        """

        os.makedirs('pdb_structures', exist_ok=True)
        os.makedirs('alphafold_structures', exist_ok=True)
        os.makedirs('membranome_structures', exist_ok=True)

        if uniprot_id is None:
            raise ValueError("Uniprot_id must be provided")
        
        # check cache
        cache_key = self._get_structure_cache_key(uniprot_id, pdb_id, structure_source)
        cached_path = self._get_cached_data(cache_key, "structure")
        if cached_path and self._check_file_exists(cached_path):
            print(f"Using cached structure: {cached_path}")
            return cached_path
        
        if pdb_id is not None:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_path = os.path.join('pdb_structures', f"{pdb_id}.pdb")
            
            # check if the file already exists
            if self._check_file_exists(pdb_path):
                print(f"PDB structure already exists: {pdb_path}")
                fixed_path = StructureFormatter.fix_pdb_numbering(pdb_path, uniprot_id)
                self._cache_data(cache_key, fixed_path, "structure")
                return fixed_path
            
            pdb_response = requests.get(pdb_url)
            if pdb_response.status_code == 200:
                with open(pdb_path, 'wb') as f:
                    f.write(pdb_response.content)
                print(f"Downloaded PDB structure for {pdb_id}: {pdb_path}")
                fixed_path = StructureFormatter.fix_pdb_numbering(pdb_path, uniprot_id)
                self._cache_data(cache_key, fixed_path, "structure")
                return fixed_path
            else:
                print(f"Error downloading PDB structure for {pdb_id}: {pdb_response.status_code}")
                return None
        
        if structure_source=='PDB': 
            pdb_mapping_url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.txt" 
            try:
                response = requests.get(pdb_mapping_url)
                if response.status_code == 200:
                    pdb_entries = []
                    for line in response.text.split('\n'):
                        if line.startswith('DR   PDB;'):
                            pdb_id = line.split(';')[1].strip()
                            pdb_entries.append(pdb_id)
                    
                    if pdb_entries:
                        pdb_id = pdb_entries[0]
                        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                        pdb_path = os.path.join('pdb_structures', f"{pdb_id}_{uniprot_id}.pdb")
                        
                        # check if the file already exists
                        if self._check_file_exists(pdb_path):
                            print(f"PDB structure already exists: {pdb_path}")
                            fixed_path = StructureFormatter.fix_pdb_numbering(pdb_path, uniprot_id)
                            self._cache_data(cache_key, fixed_path, "structure")
                            return fixed_path
                        
                        pdb_response = requests.get(pdb_url)
                        if pdb_response.status_code == 200:
                            with open(pdb_path, 'wb') as f:
                                f.write(pdb_response.content)
                            print(f"Downloaded first PDB structure for {uniprot_id}: {pdb_path}")
                            fixed_path = StructureFormatter.fix_pdb_numbering(pdb_path, uniprot_id)
                            self._cache_data(cache_key, fixed_path, "structure")
                            return fixed_path
                        else:
                            print(f"Error downloading PDB structure for {uniprot_id}: {pdb_response.status_code}")
                            return None
            except Exception as e:
                print(f"Error checking PDB for {uniprot_id}: {e}")
                return None
                
        if structure_source=='alphafold': 
            alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            alphafold_path = os.path.join('alphafold_structures', f"{uniprot_id}_alphafold.pdb")

            # check if the file already exists
            if self._check_file_exists(alphafold_path):
                print(f"AlphaFold structure already exists: {alphafold_path}")
                self._cache_data(cache_key, alphafold_path, "structure")
                return alphafold_path

            try:
                response = requests.get(alphafold_url)
                if response.status_code == 200:
                    with open(alphafold_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded AlphaFold structure for {uniprot_id}: {alphafold_path}")
                    self._cache_data(cache_key, alphafold_path, "structure")
                    return alphafold_path
                else:
                    print(f"Error downloading from AlphaFold for {uniprot_id}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading from AlphaFold for {uniprot_id}: {e}")
                
        if structure_source=='membranome':
            def get_membranome_id_from_uniprot(uniprot_id):
                # Get the UniProt entry name (e.g., TNFA_HUMAN)
                uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
                response = requests.get(uniprot_url)

                if response.status_code == 200:
                    data = response.json()
                    # Get the UniProt entry name
                    membranome_id = data.get('uniProtkbId')  # This will get "TNFA_HUMAN"
                    return membranome_id
                return None
            
            membranome_id = get_membranome_id_from_uniprot(uniprot_id)
            if not membranome_id:
                print(f"Could not get membranome ID for {uniprot_id}")
                return None
                
            membranome_url = f"https://storage.googleapis.com/membranome-assets/pdb_files/proteins/{membranome_id}.pdb"
            membranome_path = os.path.join('membranome_structures', f"{uniprot_id}_membranome.pdb")
            adjusted_file = f"{membranome_path.rsplit('.', 1)[0]}_adjusted.pdb"
            
            # check if the adjusted file already exists
            if self._check_file_exists(adjusted_file):
                print(f"Membranome structure already exists: {adjusted_file}")
                self._cache_data(cache_key, adjusted_file, "structure")
                return adjusted_file
            
            try:
                response = requests.get(membranome_url)
                if response.status_code == 200:
                    with open(membranome_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded Membranome structure for {uniprot_id}: {membranome_path}")
                    self.formatter.format_pdb(membranome_path, adjusted_file)
                    self._cache_data(cache_key, adjusted_file, "structure")
                    return adjusted_file
                else:
                    print(f"Error downloading from Membranome for {uniprot_id}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading from Membranome for {uniprot_id}: {e}")
        
        print(f"No structure found for {uniprot_id}")
        return None

    def batch_download_structures(self, uniprot_ids):

        """
        Download structures for a list of uniprot ids
        ______________
        Args:
            uniprot_ids: List of uniprot ids
        ______________
        Returns:
            dict: Mapping of uniprot id to downloaded structure path
        """

        download_results = {}
        
        if isinstance(uniprot_ids, pd.Series):
            uniprot_ids = uniprot_ids.tolist()
        
        for uniprot_id in uniprot_ids:
            structure_path = self.download_pdb_structure(uniprot_id)
            download_results[uniprot_id] = structure_path
        
        return download_results

    def read_pdb_by_source(self, structure_source, uniprot_id,pdb_id=None,chain=None):
        """
        Read the PDB structure from the source
        ______________
        Args:
            structure_source: Structure source ["membranome", "alphafold", "pdb"]
            uniprot_id: UniProt ID
            pdb_id: PDB ID
            chain: Chain identifier
        ______________
        Returns:
            valid_chains: List of valid chains
            filtered_pdb: PDB data with only the valid chains
        """
        # Get the PDB structure
        if structure_source in ["membranome", "alphafold"]:
            pdb_path = self.download_pdb_structure(uniprot_id=uniprot_id, structure_source=structure_source)
        elif structure_source == "pdb":
            pdb_path = self.download_pdb_structure(uniprot_id=uniprot_id, pdb_id=pdb_id, structure_source=structure_source)

        if not pdb_path:
            raise ValueError(f"No PDB structure found for {uniprot_id} from {structure_source}")

        with open(pdb_path, 'r') as f:
            original_pdb = f.read()

        # Get actual chain identifiers from PDB data
        existing_chains = self.parser._get_chain_identifiers(original_pdb)
        print(f"Chains in the file: {existing_chains}")
        
        # Process chain parameter logic
        if structure_source == "pdb":
            if chain:
                specified_chains = list(chain.upper().strip())
                valid_chains = [c for c in existing_chains if c.upper() in specified_chains]
                if not valid_chains:
                    raise ValueError(f"Specified chain {chain} does not exist for {uniprot_id}, available chains: {existing_chains}")
            else:
                valid_chains = existing_chains
        else:
            valid_chains = existing_chains

        # Filter PDB data
        filtered_pdb = self.parser._filter_pdb_by_chain(original_pdb, valid_chains)

        return valid_chains, filtered_pdb

    def get_prioritized_pdb_list(self, uniprot_id: str) -> List[Dict]:
        """
        Get prioritized list of available PDB structures (metadata only, no download)
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            List of PDB metadata dictionaries, sorted by priority (resolution + estimated coverage)
        """
        cache_key = self._get_structure_cache_key(uniprot_id, None, "pdb_list")
        # Check cache first
        cached_data = self._get_cached_data(cache_key, "structure")
        if cached_data:
            print(f"✓ Loaded PDB list for {uniprot_id} from cache")
            return cached_data
        
        print(f"Fetching available PDB structures for {uniprot_id}...")
        
        pdb_mapping_url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.txt"
        
        try:
            response = requests.get(pdb_mapping_url)
            if response.status_code == 200:
                pdb_entries = []
                for line in response.text.split('\n'):
                    if line.startswith('DR   PDB;'):
                        # Parse PDB entry with chain and range information
                        parts = line.split(';')
                        if len(parts) >= 4:
                            pdb_id = parts[1].strip()
                            chains = parts[2].strip() if parts[2].strip() != '-' else None
                            resolution_str = parts[3].strip() if parts[3].strip() != '-' else None
                            
                            # Parse resolution
                            resolution = None
                            if resolution_str and resolution_str != '-':
                                try:
                                    # Extract numerical value from resolution string (e.g., "2.50 A")
                                    res_match = re.search(r'(\d+\.?\d*)', resolution_str)
                                    if res_match:
                                        resolution = float(res_match.group(1))
                                except:
                                    resolution = None
                            
                            pdb_entries.append({
                                'pdb_id': pdb_id,
                                'chains': chains,
                                'resolution': resolution,
                                'resolution_str': resolution_str
                            })
                
                print(f"Found {len(pdb_entries)} available PDB structures for {uniprot_id}")
                
                # Prioritize by resolution (lower is better)
                def priority_score(entry):
                    resolution = entry['resolution']
                    if resolution is not None:
                        return 1.0 / (resolution + 0.1)  # Higher score = higher priority
                    else:
                        return 0.1  # Low priority for structures without resolution
                
                # Sort by priority score (descending = best first)
                prioritized_entries = sorted(pdb_entries, key=priority_score, reverse=True)
                
                print(f"Prioritized PDB structures:")
                for i, entry in enumerate(prioritized_entries[:5]):  # Show top 5
                    print(f"  {i+1}. {entry['pdb_id']}: resolution={entry['resolution_str']}")
                if len(prioritized_entries) > 5:
                    print(f"  ... and {len(prioritized_entries) - 5} more")
                
                # Cache the results
                self._cache_data(cache_key, prioritized_entries, "structure")
                
                return prioritized_entries
                
            else:
                print(f"Error fetching UniProt data for {uniprot_id}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error processing PDB structures for {uniprot_id}: {str(e)}")
            return []
    
    def download_and_analyze_pdb(self, uniprot_id: str, pdb_id: str) -> Dict:
        """
        Download and analyze a single PDB structure on-demand
        
        Args:
            uniprot_id: UniProt ID
            pdb_id: PDB ID to download
            
        Returns:
            Dict with structure info and coverage analysis, or None if failed
        """
        # Check cache first
        cache_key = self._get_structure_cache_key(uniprot_id, pdb_id, "pdb_analysis")
        cached_info = self._get_cached_data(cache_key, "structure")
        if cached_info and isinstance(cached_info, dict) and 'pdb_path' in cached_info:
            # Verify cached file still exists
            if self._check_file_exists(cached_info['pdb_path']):
                print(f"    ✓ Using cached analysis for {pdb_id}")
                return cached_info
        
        try:
            # Download the PDB structure
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_path = os.path.join('pdb_structures', f"{pdb_id}_{uniprot_id}.pdb")
            
            # Check if file already exists
            if not self._check_file_exists(pdb_path):
                print(f"    Downloading {pdb_id}...")
                
                pdb_response = requests.get(pdb_url)
                if pdb_response.status_code == 200:
                    os.makedirs('pdb_structures', exist_ok=True)
                    with open(pdb_path, 'wb') as f:
                        f.write(pdb_response.content)
                    print(f"    ✓ Downloaded {pdb_id}")
                else:
                    print(f"    ✗ Failed to download {pdb_id}: HTTP {pdb_response.status_code}")
                    return None
            else:
                print(f"    ✓ Using cached {pdb_id}")
            
            # Fix numbering and analyze coverage
            fixed_path = StructureFormatter.fix_pdb_numbering(pdb_path, uniprot_id)
            coverage_info = self._analyze_pdb_coverage(fixed_path, uniprot_id)
            
            if coverage_info['total_residues'] > 0:
                structure_info = {
                    'pdb_id': pdb_id,
                    'pdb_path': fixed_path,
                    'coverage_range': coverage_info['coverage_range'],
                    'covered_positions': coverage_info['covered_positions'],
                    'total_residues': coverage_info['total_residues']
                }
                
                # Cache the analysis result
                self._cache_data(cache_key, structure_info, "structure")
                
                range_start, range_end = coverage_info['coverage_range']
                print(f"    ✓ {pdb_id}: covers {range_start}-{range_end} ({coverage_info['total_residues']} residues)")
                return structure_info
            else:
                print(f"    ✗ {pdb_id}: no valid residues found")
                return None
                
        except Exception as e:
            print(f"    ✗ Error processing {pdb_id}: {str(e)}")
            return None
    
    def _analyze_pdb_coverage(self, pdb_path: str, uniprot_id: str) -> Dict:
        """
        Analyze sequence coverage of a PDB structure
        
        Args:
            pdb_path: Path to PDB file
            uniprot_id: UniProt ID for validation
            
        Returns:
            Dict with coverage information
        """

        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            
            all_positions = []
            
            # Get all residue positions from all chains
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':  # Only standard residues
                            pos = residue.id[1]
                            if isinstance(pos, int) and pos > 0:
                                all_positions.append(pos)
            
            if not all_positions:
                return {
                    'coverage_range': (None, None),
                    'covered_positions': [],
                    'total_residues': 0
                }
            
            covered_positions = sorted(list(set(all_positions)))
            
            coverage_info = {
                'coverage_range': (min(covered_positions), max(covered_positions)),
                'covered_positions': covered_positions,
                'total_residues': len(covered_positions)
            }
            
            return coverage_info
            
        except Exception as e:
            print(f"Error analyzing PDB coverage for {pdb_path}: {str(e)}")
            return {
                'coverage_range': (None, None),
                'covered_positions': [],
                'total_residues': 0
            }













