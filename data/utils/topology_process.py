from .format_process import FormatProcess

class TopologyProcess:

    @staticmethod
    def extract_topology_information(df):
        """
        Extract topology information based on dataset name for each row.
        Creates standardized columns for extracellular, transmembrane, and intracellular regions.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing topology data and dataset column
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with three new columns: 'extracellular', 'transmembrane', 'intracellular'
        """
        result_df = df.copy()
        result_df.loc[:, 'extracellular'] = None
        result_df.loc[:, 'transmembrane'] = None 
        result_df.loc[:, 'intracellular'] = None
        
        for index, row in result_df.iterrows():
            dataset_name = row['dataset']

            if 'uniprot' in dataset_name:
                result_df.at[index, 'extracellular'] = row.get('uniprot_extracellular')
                result_df.at[index, 'transmembrane'] = row.get('uniprot_transmembrane')
                result_df.at[index, 'intracellular'] = row.get('uniprot_cytoplasmic')
            elif 'tmhmm' in dataset_name:
                result_df.at[index, 'extracellular'] = row.get('tmhmm_extracellular')
                result_df.at[index, 'transmembrane'] = row.get('tmhmm_transmembrane')
                result_df.at[index, 'intracellular'] = row.get('tmhmm_cytoplasmic')
        
        for col in ['extracellular', 'transmembrane', 'intracellular']:
            result_df.loc[:, col] = result_df[col].apply(FormatProcess.parse_domain_ranges)

        return result_df