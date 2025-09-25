import pandas as pd
import numpy as np
import aaanalysis as aa
import matplotlib.pyplot as plt

class CPPProfiler:
    def __init__(self):
        pass
    
    def _get_scales(self, reduce_scales:bool=True, n_clusters:int=133):
        # Load scales
        df_scales = aa.load_scales() # 586 scales in total
        if reduce_scales:
            # Obtain redundancy-reduced set of 133 scales (need to be changed)
            aac = aa.AAclust()
            X = np.array(df_scales).T
            scales = aac.fit(X, names=list(df_scales), n_clusters=n_clusters).medoid_names_
            df_scales = df_scales[scales]
        return df_scales
    
    def get_features(self, df: pd.DataFrame, 
                            sequence_col: str = 'sequence', label_col: str = 'known_cleavage_site',
                            n_filter: int = 100,
                            accept_gaps: bool = True,
                            reduce_scales: bool = True,
                            n_clusters: int = 133):
        df_scales = self._get_scales(reduce_scales=reduce_scales, n_clusters=n_clusters)
        df_parts = pd.DataFrame({'tmd': df[sequence_col]})
        # 1 for cleavage site (test), 0 for non-cleavage site (ref)
        cpp_labels = [1 if x else 0 for x in df[label_col].to_list()]
        cpp = aa.CPP(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        df_feat = cpp.run(labels=cpp_labels,n_filter=n_filter)

        sf = aa.SequenceFeature()
        features = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"],accept_gaps=accept_gaps)
        return features, cpp_labels, df_feat
    
    def plot_cpp_profile(self, features: pd.DataFrame, cpp_labels: list, df_feat: pd.DataFrame):
        tm = aa.TreeModel()
        tm.fit(features, labels=cpp_labels)
        df_feat_new = tm.add_feat_importance(df_feat=df_feat)

        # Plot CPP ranking
        cpp_plot = aa.CPPPlot(jmd_n_len=0, jmd_c_len=0, accept_gaps=True)
        aa.plot_settings(short_ticks=True, weight_bold=False)
        cpp_plot.ranking(df_feat=df_feat_new)
        plt.show()

        aa.plot_settings(font_scale=0.9)
        cpp_plot.profile(df_feat=df_feat_new)
        plt.show()

        aa.plot_settings(font_scale=0.6, weight_bold=False)
        cpp_plot.feature_map(df_feat=df_feat_new)
        plt.show()

    def profile_windows(self, df_feat: pd.DataFrame, df: pd.DataFrame, sequence_col: str = 'sequence'):
        '''Use the given df_feat to profile the windows in df'''
        df_parts = pd.DataFrame({'tmd': df[sequence_col]})
        sf = aa.SequenceFeature()
        features = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"],accept_gaps=True)
        print(features.shape)
        return features