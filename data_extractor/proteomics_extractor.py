import os
import requests
from pyteomics import pepxml, auxiliary
from typing import List, Dict

class ProteomicsExtractor:
    def __init__(self, url: str, temp_dir: str):
        """initialize the proteomics data extractor
        
        Args:
            url: URL of the proteomics data file
            temp_dir: directory path to save the temporary file
        """
        self.url = url
        self.temp_dir = temp_dir
        
        # ensure the temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def extract_pepxml_data(self) -> str:
        """extract the data from the pepXML file
        
        Returns:
            pepXML file inside the temp directory and the path of the file
        """
        # download the pepXML file
        response = requests.get(self.url)
        # save the file to the temp directory
        temp_file_path = os.path.join(self.temp_dir, 'temp.pep.xml')
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        #with pepxml.read(temp_file_path) as reader:
            #print('Tree structure:')
            #auxiliary.print_tree(next(reader))
        df = pepxml.DataFrame(temp_file_path)
        return temp_file_path, df



