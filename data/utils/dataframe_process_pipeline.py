from data.utils.topology_process import TopologyProcess
from data.utils.format_process import FormatProcess

def process_type_I_TMP_ADAM17(df_sub, df_nonsub, df_ref):

    '''
    Prepare the df_seq to generate the CPP parts.
    '''

    '''
    sub:
    '''
    # filter the dataset (e.g. for type I)
    type_I_sub_datasets = [
        'type_I_TMP_uniprot_ADAM17_sub',
        'type_I_TMP_tmhmm_ADAM17_sub'
    ]
    df_filtered = FormatProcess.filter_dataset(df_sub, type_I_sub_datasets)
    # extract the topology information
    df_type_I_sub = TopologyProcess.extract_topology_information(df_filtered)
    # process the cleavage site information
    df_type_I_sub['final_cleavage_site'] = FormatProcess.clean_column(df_type_I_sub['final_cleavage_site'])

    '''
    nonsub:
    '''
    # filter the dataset (e.g. for type I)
    type_I_nonsub_datasets = [
        'type_I_TMP_uniprot_ADAM17_nonsub',
        'type_I_TMP_tmhmm_ADAM17_nonsub'
    ]
    df_filtered = FormatProcess.filter_dataset(df_nonsub, type_I_nonsub_datasets)
    # extract the topology information
    df_type_I_nonsub = TopologyProcess.extract_topology_information(df_filtered)

    '''
    ref:
    '''
    type_I_ref_datasets = ['type_I_TMP_uniprot_ADAM17_ref']
    df_filtered = FormatProcess.filter_dataset(df_ref, type_I_ref_datasets)
    # extract the topology information
    df_type_I_ref = TopologyProcess.extract_topology_information(df_filtered)

    '''
    Obtain ref entries that are not in sub nor nonsub
    '''
    nonsub_entries = set(df_type_I_nonsub['final_entry'])
    sub_entries = set(df_type_I_sub['final_entry'])
    all_excluded_entries = nonsub_entries.union(sub_entries)
    # filter the ref dataset
    df_type_I_ref_filtered = df_type_I_ref[~df_type_I_ref['Entry'].isin(all_excluded_entries)]

    return df_type_I_sub, df_type_I_nonsub, df_type_I_ref_filtered


def process_all_type_ADAM17(df_sub, df_nonsub, df_ref):
    pass