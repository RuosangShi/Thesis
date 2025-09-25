import requests
import pandas as pd
from typing import Dict, Any, List, Optional    
from Bio import SeqIO
from io import StringIO
from Bio.Align import substitution_matrices
from Bio.Align import PairwiseAligner
import os
import hashlib


class UniprotExtractor:
    """Fetch the protein information based on UniProt"""
    
    def __init__(self, uniprot_id: str, add_cache: bool = False, cache_dir: str = "./uniprot_cache"):
        """Initiate Uniprot extractor:
        
        Args:
            uniprot_id: UniProt ID (e.g. 'P01034')
            add_cache: bool, whether to use disk cache for FASTA sequences (default: False)
            cache_dir: str, directory to store cache files (default: './uniprot_cache')
        """
        self.uniprot_id = uniprot_id
        self.add_cache = add_cache
        self.cache_dir = cache_dir
        self.url = f"https://rest.uniprot.org/uniprotkb/{self.uniprot_id}.json"
        
        # Create cache directory if using cache
        if self.add_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.data = self.fetch_data()
        self.fasta_sequence = self.fetch_fasta_sequence()
        self.gene_sequence = self.get_gene_sequence()
        
    def fetch_data(self) -> Dict[str, Any]:
        """Obtain the protein information from UniProt with caching support
        
        Returns:
            A dictionary containing the protein information
        """
        # Generate cache key for JSON data
        cache_key = self._get_cache_key("json")
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            print(f"✓ Loaded UniProt data for {self.uniprot_id} from cache")
            import json
            return json.loads(cached_data)

        # Fetch from UniProt API
        print(f"Fetching UniProt data for {self.uniprot_id} from UniProt...")
        response = requests.get(self.url)
        if response.status_code == 200:
            data = response.json()
            
            # Save to cache if caching is enabled
            if self.add_cache:
                import json
                json_data = json.dumps(data, ensure_ascii=False, indent=2)
                self._save_to_cache(cache_key, json_data)
                print(f"✓ Cached UniProt data for {self.uniprot_id}")
            
            return data
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")
    
    def fetch_organism(self) -> Optional[str]:
        """Obtain the organism of the protein
        
        Returns:
            The organism of the protein, None if not exists
        """
        try:
            return self.data.get('organism', {}).get('commonName').upper()
        except Exception as e:
            print(f"Error fetching organism: {str(e)}")
        
        return None
    
    def fetch_name(self) -> Optional[str]:
        """Obtain the name of the protein
        
        Returns:
            The name of the protein, None if not exists
        """
        try:
            return self.data.get('uniProtkbId')
        except Exception as e:
            print(f"Error fetching name: {str(e)}")
        
        return None
    
    def fetch_gene_name(self) -> Optional[str]:
        """Obtain the gene name
        
        Returns:
            The gene name, None if not exists
        """
        try:
            for gene in self.data.get('genes', []):
                if 'geneName' in gene:
                    return gene['geneName'].get('value')
        except Exception as e:
            print(f"Error fetching gene name: {str(e)}")
        
        return None

    def _get_cache_key(self, data_type: str) -> str:
        """
        Generate cache key for the given data type and UniProt ID
        
        Args:
            data_type: Type of data being cached (e.g., 'fasta')
            
        Returns:
            str: Hash-based cache key
        """
        cache_string = f"{self.uniprot_id}_{data_type}"
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
                    return f.read()
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

    def fetch_fasta_sequence(self):
        """
        Retrieve full FASTA sequence by UniProt ID with optional disk caching
        
        Returns:
            str: Complete FASTA sequence string
            (e.g. '>sp|P05067|HSP_HUMAN Heat shock protein 70 kDa'
            'MALWMRLLPLLALLALWGPDPAAA...')
        """
        # Generate cache key
        cache_key = self._get_cache_key("fasta")
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            print(f"✓ Loaded FASTA sequence for {self.uniprot_id} from cache")
            return cached_data

        # Fetch from UniProt API
        print(f"Fetching FASTA sequence for {self.uniprot_id} from UniProt...")
        url = f"https://rest.uniprot.org/uniprotkb/{self.uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.status_code == 200:
            fasta_data = response.text
            
            # Save to cache if caching is enabled
            self._save_to_cache(cache_key, fasta_data)
            if self.add_cache:
                print(f"✓ Cached FASTA sequence for {self.uniprot_id}")
            
            return fasta_data
        else:
            raise ValueError(f"Failed to fetch FASTA sequence for UniProt ID {self.uniprot_id}, HTTP status: {response.status_code}")


    def get_gene_sequence(self):
        """Process FASTA string to get sequence.
        
        Returns:
            The gene sequence, None if not exists
            (e.g. 'MALWMRLLPLLALLALWGPDPAAA...')
        """
        fasta = StringIO(self.fasta_sequence)
        for record in SeqIO.parse(fasta, "fasta"):
            return str(record.seq)
        return None


    def fetch_protein_type(self) -> List[str]:
        """Obtain the protein type
        
        Returns:
            A list of protein types, e.g. ['Type I transmembrane protein']
        """
        result = []
        for comment in self.data.get('comments', []):
            if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                for location in comment.get('subcellularLocations', []): 
                    if 'topology' in location:
                        topo_value = location['topology'].get('value', '')
                        result.append(topo_value)
        return list(set(result))
    
    def fetch_protein_location(self) -> List[str]:
        """Obtain the protein location
        
        Returns:
            A list of protein locations, e.g. ['Extracellular']
        """
        result = []
        for comment in self.data.get('comments', []):
            if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                for location in comment.get('subcellularLocations', []):
                    # Get location
                    if 'location' in location:
                        loc_value = location['location'].get('value', '')
                        result.append(loc_value)
        return list(set(result))
    
    def fetch_gene_topology(self) -> Dict[str, Any]:
        """
        Fetch detailed topology and subcellular location information for a protein.

        Returns:
            A dictionary containing the protein topology and subcellular location information
            (e.g. 'Transmembrane': ['start-end', 'start-end'])
        """
        result = {
                'Transmembrane': [],
                'Extracellular': [],
                'Cytoplasmic': []
            }
        for feature in self.data.get('features', []):
            if feature.get('type') == 'Transmembrane':
                start = feature['location']['start']['value']
                end = feature['location']['end']['value']
                result['Transmembrane'].append(f"{start}-{end}")
            elif feature.get('type') == 'Topological domain':
                description = feature.get('description', '').lower()
                start = feature['location']['start']['value']
                end = feature['location']['end']['value']
                if 'extracellular' in description:
                    result['Extracellular'].append(f"{start}-{end}")
                elif 'cytoplasmic' in description:
                    result['Cytoplasmic'].append(f"{start}-{end}")
            
        convert = ['Transmembrane','Extracellular','Cytoplasmic']
        for entry in convert:
            result[entry] = ', '.join(result[entry])

        return result
    
    def fetch_GO_terms(self) -> Dict[str, List[Dict[str, str]]]:
        pass
    
    def clear_cache(self, data_type: str = None) -> None:
        """
        Clear cache files
        
        Args:
            data_type: Specific data type to clear (e.g., 'fasta'). 
                      If None, clears all cache files for this UniProt ID
        """
        if not self.add_cache:
            print("Caching is not enabled")
            return
            
        if data_type:
            # Clear specific cache type
            cache_key = self._get_cache_key(data_type)
            cache_file = self._get_cache_file_path(cache_key)
            
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"✓ Cleared {data_type} cache for {self.uniprot_id}")
                except Exception as e:
                    print(f"Warning: Failed to remove cache file {cache_file}: {e}")
            else:
                print(f"No {data_type} cache found for {self.uniprot_id}")
        else:
            # Clear all cache files for this UniProt ID
            cleared_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(self.uniprot_id) and filename.endswith('.txt'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                        cleared_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to remove cache file {filename}: {e}")
            
            if cleared_count > 0:
                print(f"✓ Cleared {cleared_count} cache files for {self.uniprot_id}")
            else:
                print(f"No cache files found for {self.uniprot_id}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data
        
        Returns:
            Dict containing cache status and file info
        """
        cache_info = {
            'cache_enabled': self.add_cache,
            'cache_dir': self.cache_dir,
            'uniprot_id': self.uniprot_id,
            'cached_files': []
        }
        
        if not self.add_cache:
            return cache_info
            
        if not os.path.exists(self.cache_dir):
            return cache_info
            
        # Check for FASTA cache
        fasta_key = self._get_cache_key("fasta")
        fasta_file = self._get_cache_file_path(fasta_key)
        
        if os.path.exists(fasta_file):
            try:
                file_size = os.path.getsize(fasta_file)
                cache_info['cached_files'].append({
                    'type': 'fasta',
                    'file': fasta_file,
                    'size_bytes': file_size,
                    'size_kb': round(file_size / 1024, 2)
                })
            except Exception as e:
                print(f"Warning: Failed to get file info for {fasta_file}: {e}")
        
        # Check for JSON cache
        json_key = self._get_cache_key("json")
        json_file = self._get_cache_file_path(json_key)
        
        if os.path.exists(json_file):
            try:
                file_size = os.path.getsize(json_file)
                cache_info['cached_files'].append({
                    'type': 'json',
                    'file': json_file,
                    'size_bytes': file_size,
                    'size_kb': round(file_size / 1024, 2)
                })
            except Exception as e:
                print(f"Warning: Failed to get file info for {json_file}: {e}")
        
        return cache_info
    

    def get_all_info(self) -> Dict[str, Any]:
        """Obtain the comprehensive report of all information
        
        Returns:
            A dictionary containing all the extracted information
        """
        return {
            'uniprot_id': self.uniprot_id,
            'organism': self.fetch_organism(),
            'name': self.fetch_name(),
            'gene_name': self.fetch_gene_name(),
            'sequence': self.get_gene_sequence(),
            'protein_type': self.fetch_protein_type(),
            'protein_location': self.fetch_protein_location(),
            'topology': self.fetch_gene_topology(),
        }
    
    def match_mouse_human_uniprot(self, threshold: float = 70.0) -> Dict[str, Any]:
        """
        For a given uniprot ID, match the corresbonding mouse or human uniprot ID

        Args:
            threshold: Identity threshold (default: 70%)
            
        Returns:
            A dictionary containing the mouse and human uniprot ID, 
            sequence alignment info and identity percentage
        """
        # Determine the current organism and the target organism
        organism = self.fetch_organism()
        current_sequence = self.get_gene_sequence()
        
        if "HUMAN" in organism:
            source_organism = "human"
            target_organism = "mouse"
            target_taxid = "10090"  # Mouse taxon ID
        elif "MOUSE" in organism:
            source_organism = "mouse"
            target_organism = "human"
            target_taxid = "9606"   # Human taxon ID
        else:
            return {"error": f"Unsupported organism: {organism}"}
        
        # Get orthologs using UniProt API
        url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{self.fetch_gene_name()}+AND+organism_id:{target_taxid}&format=json"
        response = requests.get(url)
        
        if response.status_code != 200:
            return {"error": f"Failed to fetch ortholog data: {response.status_code}"}
        ortholog_data = response.json()
        
        if not ortholog_data.get("results"):
            return {"error": f"No {target_organism} ortholog found for {self.uniprot_id}"}
        
        # Get the best match (first result)
        target_uniprot_id = ortholog_data["results"][0]["primaryAccession"]
        # Fetch the target protein sequence
        target_url = f"https://rest.uniprot.org/uniprotkb/{target_uniprot_id}.fasta"
        target_response = requests.get(target_url)
        if target_response.status_code != 200:
            return {"error": f"Failed to fetch target sequence for {target_uniprot_id}"}
        target_fasta = target_response.text
        target_sequence = None
        fasta = StringIO(target_fasta)
        for record in SeqIO.parse(fasta, "fasta"):
            target_sequence = str(record.seq)
            break
        if not target_sequence:
            return {"error": "Could not parse target sequence"}
        
        # Perform sequence alignment (using simple identity calculation)
        alignment_result = self._calculate_sequence_identity_with_blosum62(current_sequence, target_sequence)
        print(f"alignment_result: {alignment_result}")
        
        return pd.Series({
            "source_organism": source_organism,
            "source_uniprot_id": self.uniprot_id,
            "target_organism": target_organism,
            "target_uniprot_id": target_uniprot_id,
            "score_percentage": alignment_result["score_percentage"],
            "homology_level": alignment_result["homology_level"],
            "score_threshold_passed": alignment_result["score_percentage"] >= threshold  # Default threshold of 70%
        })
    
    def _calculate_sequence_identity_with_blosum62(self, seq1: str, seq2: str) -> Dict[str, Any]:
        try:
            # load BLOSUM62 matrix
            blosum62 = substitution_matrices.load("BLOSUM62")
            
            # get the alphabet and matrix
            alphabet = blosum62.alphabet
            
            # calculate the maximum possible score
            max_possible_score = 0
            for aa in seq1:
                try:
                    # get the index of the amino acid in the alphabet
                    aa_index = alphabet.find(aa)
                    if aa_index >= 0:
                        # use the index to get the score from the matrix
                        max_possible_score += blosum62[aa_index, aa_index]
                except Exception as e:
                    print(f"Error processing amino acid {aa}: {str(e)}")
                    continue
            
            # set the aligner
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            #no gap penalty
            aligner.substitution_matrix = blosum62
            alignment = aligner.align(seq1, seq2)[0]
            for aa in seq1:
                if aa in blosum62:
                    max_possible_score += blosum62[aa, aa]
            
            # calculate the score percentage
            score_percentage = (alignment.score / max_possible_score) * 100 if max_possible_score > 0 else 0
            
            # set the threshold standard based on BLOSUM62
            # 30-50%: distal homology
            # 50-70%: medium homology
            # >70%: proximal homology
            if 30 <= score_percentage < 50:
                homology_level = "distal homology"
            elif 50 <= score_percentage < 70:
                homology_level = "medium homology"
            elif score_percentage >= 70:
                homology_level = "proximal homology"
            else:
                homology_level = "non-homology"

            return {
                "score_percentage": round(score_percentage, 2),
                "homology_level": homology_level
            }
        except Exception as e:
            print(f"BLOSUM62 scoring error: {str(e)}")
            return {"error": f"Scoring failed: {str(e)}"}
        
        