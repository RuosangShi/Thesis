import requests
from typing import List

def get_pdb_ids(uniprot_id: str) -> List[str]:
    """
    Get all corresponding PDB IDs from UniProt ID
    
    Parameters:
        uniprot_id: UniProt accession ID (e.g. "P12821")
        
    Returns:
        A list of PDB IDs sorted by appearance order
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    params = {
        'fields': 'xref_pdb',  # Request only PDB-related fields
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        pdb_entries = data.get('uniProtKBCrossReferences', [])
        
        # Filter and extract PDB IDs
        return [
            entry['id'] for entry in pdb_entries 
            if entry.get('database') == 'PDB'
        ]
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {str(e)}")
        return []
    except KeyError:
        print("Data format exception")
        return []
    except Exception as e:
        print(f"Unknown error: {str(e)}")
        return []
