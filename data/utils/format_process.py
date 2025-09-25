import pandas as pd
import numpy as np
import ast
from data_extractor.uniprot_extractor import UniprotExtractor

class FormatProcess:
    
    @staticmethod
    def clean_column(column):
        """
        !! designed for cleavage site !!

        Ensure a column contains lists of integers.
        Convert strings to lists if possible, and handle NaN or invalid values.
        e.g. "[145, 185, 384]" to [145, 185, 384]
        """
        cleaned_column = []
        for value in column:
            try:
                if pd.isna(value):  # Handle NaN values
                    cleaned_column.append([]) 
                elif isinstance(value, str):  # String representation of a list
                    parsed_value = ast.literal_eval(value)
                    # e.g. change "[145, 185, 384]" to [145, 185, 384]
                    cleaned_column.append(parsed_value)
                else:
                    cleaned_column.append([])  # Invalid type
            except (ValueError, SyntaxError, TypeError):  # Handle malformed strings or types
                cleaned_column.append([])
        return pd.Series(cleaned_column)

    @staticmethod
    def parse_domain_ranges(domain_str):
        """
        !! designed for topology information !!
        
        Parse domain ranges from a string like '31-948, 1002-1008' into a list of tuples.
        Handle malformed or missing strings gracefully.
        e.g. '31-948, 1002-1008' to [(31, 948), (1002, 1008)]
        """
        if pd.isna(domain_str) or not isinstance(domain_str, str):
            return []
        try:
            return [tuple(map(int, r.split('-'))) for r in domain_str.split(',') if '-' in r]
        except ValueError:
            return []  # Handle malformed entries
        
    @staticmethod
    def filter_dataset(df_data, dataset_name):
        """
        !! designed for filtering dataset !!

        Process the dataset to get the data for the given dataset name.
        e.g. dataset_name = ['human_type_I_TMP_uniprot_ADAM17_sub', 
                                'mouse_type_I_TMP_uniprot_ADAM17_sub']
        """
        return df_data[df_data['dataset'].isin(dataset_name)].reset_index(drop=True)
    
    
    @staticmethod
    def process_for_df_seq(df_data, data_type, entry_col='final_entry'):
        """
        !! Designed for analysis.SequenceFeature.get_df_parts !!

        df_data should have following columns:
        entry_col: uniprot ids, default is 'final_entry' in dataset, if it is reference dataset, it should be 'Entry'
        'transmembran': transmembran regions, like [(702, 722)]	

        since only type I or type II will be considered, all protein should be single pass

        return a dataframe with the following columns:
        'sequence': The complete amino acid sequence.
        'tmd_start': Starting positions of the TMD in the sequence.
        'tmd_stop': Ending positions of the TMD in the sequence.
        'label': The label of the dataset. test=1, reference=0
        """
        df_seq = df_data.copy()
        df_seq['entry'] = df_seq[entry_col]
        df_seq['tmd_start'] = df_seq['transmembrane'].apply(lambda x: x[0][0] if x else None)
        df_seq['tmd_stop'] = df_seq['transmembrane'].apply(lambda x: x[0][1] if x else None)
        if data_type == 'nonsub':
            df_seq['label'] = 0
        elif data_type == 'sub':
            df_seq['label'] = 1
        elif data_type == 'ref':
            df_seq['label'] = 2
        elif data_type == 'dpu':
            df_seq['label'] = 3
        for index, row in df_seq.iterrows():
            try:
                extractor = UniprotExtractor(row[entry_col], add_cache=True, cache_dir='./uniprot_cache')
                df_seq.at[index, 'sequence'] = extractor.get_gene_sequence()
            except Exception as e:
                print(f"Error processing sequence for {row[entry_col]}: {str(e)}")
                df_seq.at[index, 'sequence'] = None

        df_seq = df_seq.dropna(subset=['entry', 'sequence', 'tmd_start', 'tmd_stop'], how='any')

        return df_seq[['entry', 'sequence', 'tmd_start', 'tmd_stop', 'label']]
    
    @staticmethod
    def process_for_df_emb(df_data, entry_col='final_entry'):
        """
        !! Designed for embedding !!

        df_data should have following columns:
        entry_col: uniprot ids, default is 'final_entry' in dataset, if it is reference dataset, it should be 'Entry'
        """
        df_emb = df_data.copy()
        df_emb['entry'] = df_emb[entry_col]
        df_emb['tmd_start'] = df_emb['transmembrane'].apply(lambda x: x[0][0] if x else None)
        df_emb['tmd_stop'] = df_emb['transmembrane'].apply(lambda x: x[0][1] if x else None)
        for index, row in df_emb.iterrows():
            try:
                extractor = UniprotExtractor(row[entry_col])
                df_emb.at[index, 'sequence'] = extractor.get_gene_sequence()
            except Exception as e:
                print(f"Error processing sequence for {row[entry_col]}: {str(e)}")
                df_emb.at[index, 'sequence'] = None
        df_emb = df_emb.dropna(subset=['entry', 'sequence', 'tmd_start', 'tmd_stop'], how='any')
        return df_emb[['entry', 'sequence', 'tmd_start', 'tmd_stop']]

