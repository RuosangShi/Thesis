'''
Global Feature Extraction

Global Feature Dimensions (Total per window: 1764):
- Weighted matrices (14): PWM, PSSM, NNS, KNN, and 10 SMI matrices
- CKSAAP composition (1600) 
- CPP patterns (100)
- Coevolution patterns (50)
'''
from ...weighted_metrix import PWM, PSSM, NNS, SMI, KNN
from ...cksaap import CKSAAP
from ...cpp_profiler import CPPProfiler
from ...coevo_patterens import coevo_patterns

from typing import Dict, Any
import pandas as pd
import numpy as np  


class GlobalFeatures:
    def __init__(self, training_dataframe: pd.DataFrame, 
                 include_weighted_metrix: bool = True,
                 include_cksaap: bool = True,
                 include_cpp: bool = True, 
                 cpp_n_filter: int = 100,
                 use_fimo: bool = False, fimo_windows: pd.DataFrame = None,
                 include_coevolution_patterns: bool = True):
        self.training_dataframe = training_dataframe
        self.built_profiles = {}

        self.include_weighted_metrix = include_weighted_metrix
        self.include_cksaap = include_cksaap
        self.include_cpp = include_cpp
        self.cpp_n_filter = cpp_n_filter
        self.use_fimo = use_fimo
        self.fimo_windows = fimo_windows
        self.include_coevolution_patterns = include_coevolution_patterns

        self._build_fimo_windows()
        self._build_profiles()

    def _build_fimo_windows(self):
        """Build FIMO training data if enabled."""
        if self.use_fimo:
            if self.fimo_windows is not None:
                # Use FIMO windows: combine positive training windows with FIMO windows
                positive_windows = self.training_dataframe[self.training_dataframe['known_cleavage_site'] == 1][['sequence', 'known_cleavage_site']]
                self.training_fimo = pd.concat(
                    [positive_windows, self.fimo_windows[['sequence', 'known_cleavage_site']]], 
                    ignore_index=True)
                self.training_fimo = self.training_fimo.drop_duplicates(subset=['sequence']).reset_index(drop=True)
                print(f"FIMO windows built successfully with {len(self.training_fimo)} samples")
            else:
                # No FIMO windows provided, fall back to training data
                print("Warning: use_fimo=True but fimo_windows=None. Using training data instead.")
                self.training_fimo = self.training_dataframe.copy()
        else:
            # FIMO not used, set to None
            self.training_fimo = None
    
    def _build_profiles(self):
        """Build all profiles from training data once."""
        print(f"Building global profiles from {len(self.training_dataframe)} training samples...")

        if self.include_weighted_metrix:
            # Build weighted matrix profiles
            self.built_profiles['pwm'] = PWM(pseudocount=0.01)
            self.built_profiles['pwm'].build(
                windows_df=self.training_dataframe,
                sequence_col='sequence',
                label_col='known_cleavage_site',
                self_defined_background_freq=True,
                fasta_file="data/reference_set/SwissProt_human_type_I_TMP.fasta"
            )
            
            self.built_profiles['pssm'] = PSSM(pseudocount=0.01)
            self.built_profiles['pssm'].build(
                windows_df=self.training_dataframe,
                sequence_col='sequence',
                label_col='known_cleavage_site',
                self_defined_background_freq=True,
                fasta_file="data/reference_set/SwissProt_human_type_I_TMP.fasta"
            )
            
            self.built_profiles['nns'] = NNS()
            self.built_profiles['nns'].build(
                windows_df=self.training_dataframe,
                sequence_col='sequence',
                label_col='known_cleavage_site'
            )
            
            self.built_profiles['knn'] = KNN(k=5)
            self.built_profiles['knn'].build(
                windows_df=self.training_dataframe,
                sequence_col='sequence',
                label_col='known_cleavage_site'
            )
            
            self.built_profiles['smi'] = SMI()
            self.built_profiles['smi'].build(
                windows_df=self.training_dataframe,
                sequence_col='sequence',
                label_col='known_cleavage_site'
            )
        if self.include_cksaap:
            self.built_profiles['cksaap'] = CKSAAP(k_values=[0, 1, 2, 3])
        
        if self.include_cpp:
            cpp_profiler = CPPProfiler()
            if self.use_fimo and self.training_fimo is not None:
                # Use FIMO-enhanced training data for CPP
                _, _, df_feat = cpp_profiler.get_features(self.training_fimo, n_filter = self.cpp_n_filter, reduce_scales=True, n_clusters=133)
                print(f"CPP features built using FIMO-enhanced data ({len(self.training_fimo)} samples, n_filter={self.cpp_n_filter})")
            else:
                # Use regular training data for CPP
                _, _, df_feat = cpp_profiler.get_features(self.training_dataframe, n_filter = self.cpp_n_filter, reduce_scales=True, n_clusters=133)
                print(f"CPP features built using training data ({len(self.training_dataframe)} samples, n_filter={self.cpp_n_filter})")
            self.built_profiles['cpp'] = cpp_profiler
            self.built_profiles['cpp_features'] = df_feat
        
        if self.include_coevolution_patterns:
            coevo = coevo_patterns(confidence_level=0.95, critical_value=1.96, select_top_features=True, max_features=50)
            selected_patterns, _ = coevo.extract_coevolutionary_patterns(self.training_dataframe)
            self.built_profiles['coevo'] = coevo
            self.built_profiles['coevo_patterns'] = selected_patterns
        
        print("Global profiles built successfully")
    
    def profile_features(self, profiled_dataframe: pd.DataFrame = None,
                        return_fusion_format: bool = False) -> Dict[str, Any]:
        """Apply pre-built profiles to data."""
        
        if profiled_dataframe is None:
            profiled_dataframe = self.training_dataframe
        
        features = {}
        
        # Apply profiles
        pwm_features = pssm_features = nns_features = knn_features = smi_features = None
        cksaap_df = cpp_df = coevo_df = None
        
        if self.include_weighted_metrix:
            pwm_features = self.built_profiles['pwm'].generate_score_dataframe(profiled_dataframe, include_original_cols=False)
            pssm_features = self.built_profiles['pssm'].generate_score_dataframe(profiled_dataframe, include_original_cols=False)
            nns_features = self.built_profiles['nns'].generate_score_dataframe(profiled_dataframe, include_original_cols=False)
            knn_features = self.built_profiles['knn'].generate_score_dataframe(profiled_dataframe, include_original_cols=False)
            smi_features = self.built_profiles['smi'].generate_score_dataframe(profiled_dataframe, include_original_cols=False)
            
            features['pwm_score'] = pwm_features['PWM_SCORE']
            features['pssm_score'] = pssm_features['PSSM_SCORE']
            features['nns_score'] = nns_features['NNS_SCORE']
            features['knn_score'] = knn_features['KNN_SCORE']
            
            for feature in smi_features.columns:
                if 'SMI' in feature and 'AVERAGE' not in feature:
                    features[feature.lower()] = smi_features[feature]
        
        if self.include_cksaap:
            cksaap_df = self.built_profiles['cksaap'].transform(profiled_dataframe, show_original_data=False)
            features['cksaap'] = cksaap_df
        
        if self.include_cpp:
            cpp_features = self.built_profiles['cpp'].profile_windows(self.built_profiles['cpp_features'], profiled_dataframe)
            col_names = [f'cpp_feature_{col+1}' for col in pd.DataFrame(cpp_features).columns.tolist()]
            cpp_df = pd.DataFrame(cpp_features, columns=col_names)
            features['cpp'] = cpp_df
        
        if self.include_coevolution_patterns:
            coevo_df = self.built_profiles['coevo'].profile_windows(
                profiled_dataframe,
                sequence_col='sequence',
                selected_patterns=self.built_profiles['coevo_patterns'],
                show_original_data=False
            )
            features['coevo_patterns'] = coevo_df
        
        if return_fusion_format:
            return self._to_fusion_format(pwm_features, pssm_features, nns_features, knn_features, smi_features, cksaap_df, cpp_df, coevo_df)
        
        return features
    
    def _to_fusion_format(self, pwm_df, pssm_df, nns_df, knn_df, smi_df, cksaap_df, cpp_df, coevo_df):
        """Convert to fusion format."""
        arrays = {}
        
        # Get sample count from any available DataFrame
        N = None
        for df in [pwm_df, cksaap_df, cpp_df, coevo_df]:
            if df is not None:
                N = len(df)
                break
        
        if N is None:
            raise ValueError("No feature DataFrames provided to _to_fusion_format")
        
        # Weighted features (14 dims)
        if all(df is not None for df in [pwm_df, pssm_df, nns_df, knn_df, smi_df]):
            expected_smi_cols = [
                'SMI_BLOSUM100', 'SMI_BLOSUM75', 'SMI_BLOSUM62', 'SMI_BLOSUM45', 'SMI_BLOSUM30',
                'SMI_PAM500', 'SMI_PAM400', 'SMI_PAM300', 'SMI_PAM120', 'SMI_PAM30'
            ]
            
            arrays['weighted'] = np.concatenate([
                np.asarray(pwm_df['PWM_SCORE']).reshape(N, 1),
                np.asarray(pssm_df['PSSM_SCORE']).reshape(N, 1),
                np.asarray(nns_df['NNS_SCORE']).reshape(N, 1),
                np.asarray(knn_df['KNN_SCORE']).reshape(N, 1),
                np.asarray(smi_df[expected_smi_cols]).astype(float)
            ], axis=1)
        
        # Other features
        if cksaap_df is not None:
            arrays['cksaap'] = np.asarray(cksaap_df, dtype=float)
        
        if cpp_df is not None:
            arrays['cpp'] = np.asarray(cpp_df, dtype=float)
        
        if coevo_df is not None:
            arrays['coevolution_patterns'] = np.asarray(coevo_df, dtype=float)
        
        return arrays