'''CD-HIT'''

import aaanalysis as aa
import pandas as pd

class CDHit:
    def __init__(self):
        self.sf = aa.SequenceFeature()

    def _combine_seq(self, df_seq, jmd_n_len=50, jmd_c_len=10, list_parts=['tmd','jmd_n', 'jmd_c']):
        '''
        Combine the sequence of the splited parts for CD-HIT filtering
        '''
        df_parts = self.sf.get_df_parts(df_seq=df_seq,
                           list_parts=list_parts,
                           jmd_n_len=jmd_n_len,
                           jmd_c_len=jmd_c_len).reset_index(names='entry')
        df_parts['sequence'] = df_parts[list_parts].apply(lambda x: ''.join(x), axis=1)
        return df_parts
    
    def filter_seq(self, df_seq, filter_type='partial',
                   jmd_n_len=50, jmd_c_len=10, list_parts=['tmd','jmd_n', 'jmd_c'],
                   similarity_threshold=0.4, print_info=True):
        '''
        Filter the sequence using CD-HIT.
        filter_type: 'partial' or 'full'
        '''
        if filter_type == 'partial':
            df_parts = self._combine_seq(df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, list_parts=list_parts)
        elif filter_type == 'full':
            df_parts = df_seq
        df_clust = aa.filter_seq(df_seq=df_parts, similarity_threshold=similarity_threshold)
        df_clust = df_clust[df_clust['is_representative']==1].reset_index(drop=True)
        if print_info:
            print(f"Number of entries before filtering: {df_seq.shape[0]}")
            print(f"Number of entries after filtering: {df_clust.shape[0]}")
        return df_clust
