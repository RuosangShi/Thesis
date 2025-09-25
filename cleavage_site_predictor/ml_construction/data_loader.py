#!/usr/bin/env python3
"""
Leak-Safe Feature Extractor for ML Construction Pipeline
======================================================

This module implements leak-safe feature extraction using existing node and global
feature extractors with proper fit/transform pattern to prevent data leakage.

Key Principles:
1. Feature extraction parameters are ONLY computed from (sub)training data
2. test is transformed by train (outer fold level)
3. val is transformed by subtrain (segment level)
4. No test/val data ever influences feature extraction parameters
"""

import pandas as pd
import numpy as np
import os
from contextlib import redirect_stdout
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from .feature_extractor.node_features import NodeFeatures
from .feature_extractor.global_features import GlobalFeatures
from .config_manager import get_config_manager

class DataLoader:
    """
    Leak-safe feature extractor implementing proper fit/transform pattern.
    
    This class uses the existing node and global feature extractors but ensures
    no data leakage by strictly separating parameter computation (fit) from 
    transformation (transform).
    
    Enhanced with feature configuration support for ablation studies.
    
    Usage:
        # Standard usage
        extractor = DataLoader()
        train_features = extractor.fit_transform(train_data)
        test_features = extractor.transform(test_data)
        
        # Ablation study usage
        custom_config = {'include_plm_embeddings': False, 'include_structure_features': True}
        extractor = DataLoader(feature_config=custom_config)
        train_features = extractor.fit_transform(train_data)
    """
    
    def __init__(self, config_manager=None, feature_config=None, verbose=False):
        """
        Initialize leak-safe feature extractor.
        
        Args:
            config_manager: ConfigManager instance. If None, uses global instance.
            feature_config: Custom feature configuration dict. If provided, overrides config_manager settings.
            verbose: Whether to print detailed progress information.
        """
        self.config_manager = config_manager or get_config_manager()
        
        # Use custom feature config if provided, otherwise get from config manager
        if feature_config is not None:
            self.feature_config = feature_config.copy()
            if verbose:
                print(f"   Using custom feature configuration with {len(feature_config)} settings")
        else:
            self.feature_config = self.config_manager.get_feature_config()
        
        self.data_config = self.config_manager.get_data_processing_config()

        # Fitted components (populated during fit())
        self.fitted_global_features = None  # GlobalFeatures instance fitted on training data
        self.fitted_node_features = None    # NodeFeatures instance fitted on training data
        self.scaler = None                  # Scaler fitted on training features
        self.imputer = None                 # Imputer fitted on training features
        
        # State tracking
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None

        self.verbose = verbose
        
        # Display comprehensive feature configuration
        self._print_feature_configuration()
    
    def _print_feature_configuration(self):
        """Print comprehensive feature configuration summary."""
        print("   * FEATURE CONFIGURATION:")
        print("   " + "=" * 50)
        
        # Node-level features
        print("   * Node-level Features (mean-pooled to window):")
        seq_enabled = self.feature_config.get('include_sequence_features', False)
        struct_enabled = self.feature_config.get('include_structure_features', False)
        plm_enabled = self.feature_config.get('include_plm_embeddings', False)
        
        print(f"      • Sequence features: {'✅' if seq_enabled else '❌'}")
        print(f"      • Structure features: {'✅' if struct_enabled else '❌'}")
        print(f"      • PLM embeddings: {'✅' if plm_enabled else '❌'}")
        
        if plm_enabled:
            embedding_level = self.feature_config.get('embedding_level', 'residue')
            print(f"        └─ Embedding level: {embedding_level}")
        
        # Node pooling configuration
        if seq_enabled or struct_enabled or plm_enabled:
            pooling_method = self.feature_config.get('pooling_method', 'mean')
            decay_strength = self.feature_config.get('decay_strength', 1.0)
            print(f"      • Pooling method: {pooling_method}")
            if pooling_method != 'mean':
                print(f"        └─ Decay strength: {decay_strength}")
        
        # Global features
        print("   * Global Features (per window):")
        weighted_enabled = self.feature_config.get('include_weighted_metrix', False)
        cksaap_enabled = self.feature_config.get('include_cksaap', False)
        cpp_enabled = self.feature_config.get('include_cpp', False)
        coevol_enabled = self.feature_config.get('include_coevolution_patterns', False)
        fimo_enabled = self.feature_config.get('use_fimo', False)
        
        print(f"      • Weighted matrices: {'✅' if weighted_enabled else '❌'}")
        print(f"      • CKSAAP composition: {'✅' if cksaap_enabled else '❌'}")
        print(f"      • CPP patterns: {'✅' if cpp_enabled else '❌'}")
        print(f"      • Coevolution patterns: {'✅' if coevol_enabled else '❌'}")
        print(f"      • FIMO motifs: {'🔒' if not fimo_enabled else '✅'} {'(disabled by design)' if not fimo_enabled else ''}")

        
        # Data processing options
        print("   *  Data Processing:")
        return_fusion = self.feature_config.get('return_fusion_format', True)
        normalization = self.feature_config.get('normalization', 'zscore')
        print(f"      • Fusion format: {'✅' if return_fusion else '❌'}")
        print(f"      • Normalization: {normalization}")
        
        # Summary statistics
        enabled_node_features = sum([seq_enabled, struct_enabled, plm_enabled])
        enabled_global_features = sum([weighted_enabled, cksaap_enabled, cpp_enabled, coevol_enabled, fimo_enabled])
        total_enabled = enabled_node_features + enabled_global_features
        
        print("   * Configuration Summary:")
        print(f"      • Node feature groups: {enabled_node_features}/3 enabled")
        print(f"      • Global feature groups: {enabled_global_features}/5 enabled") 
        print(f"\n      • Total feature groups: {total_enabled}/8 enabled")
        
        print("   " + "=" * 50)
    
    def _suppress_output(self):
        """Context manager to suppress output when verbose is False"""
        if self.verbose:
            from contextlib import nullcontext
            return nullcontext()
        else:
            return redirect_stdout(open(os.devnull, 'w'))
    
    def fit(self, train_data: pd.DataFrame) -> 'DataLoader':
        """
        Fit feature extraction parameters on training data ONLY.
        
        This method computes all feature extraction parameters using ONLY the
        training data to prevent leakage. All fitted parameters will be used
        for transforming validation or test data.
        
        Args:
            train_data: Training dataset to fit parameters on
            
        Returns:
            Self for method chaining
        """
        with self._suppress_output():
            print(f"   Fitting feature extraction parameters on {len(train_data)} training samples")
            
            # 1. Fit Global Features (if enabled)
            if (self.feature_config['include_weighted_metrix'] or 
                self.feature_config['include_cksaap'] or 
                self.feature_config['include_cpp'] or
                self.feature_config['include_coevolution_patterns']):
                
                print("   Fitting global feature parameters...")
                self.fitted_global_features = GlobalFeatures(
                    training_dataframe=train_data,
                    include_weighted_metrix=self.feature_config['include_weighted_metrix'],
                    include_cksaap=self.feature_config['include_cksaap'],
                    include_cpp=self.feature_config['include_cpp'],
                    cpp_n_filter=self.feature_config['cpp_n_filter'],
                    use_fimo=self.feature_config['use_fimo'],
                    fimo_windows=None, 
                    include_coevolution_patterns=self.feature_config['include_coevolution_patterns']
                )
            
            # 2. Extract training features to fit scaling parameters
            train_features = self._extract_features_internal(train_data)
            X_train = train_features['X']
            
            # 4. Fit scaler on training features only
            normalization = self.feature_config['normalization']
            if normalization == 'zscore':
                self.scaler = StandardScaler()
            elif normalization == 'minmax':
                self.scaler = MinMaxScaler()
            elif normalization == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = None
            
            if self.scaler is not None:
                X_train = self.scaler.fit_transform(X_train)
                print(f"   Fitted {normalization} scaler")
            
            # 5. Store fitted parameters
            self.feature_names = train_features.get('feature_names', 
                                                   [f'feature_{i}' for i in range(X_train.shape[1])])
            self.n_features = X_train.shape[1]
            self.is_fitted = True
            
            print("   Feature extraction parameters fitted successfully")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform data using previously fitted parameters.
        
        This method applies the feature extraction parameters computed during fit()
        to new data without recomputing any parameters, preventing data leakage.
        
        Args:
            data: Data to transform using fitted parameters
            
        Returns:
            Dictionary containing transformed features and labels
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform(). No parameters have been fitted.")
        
        with self._suppress_output():
            print(f"   Transforming {len(data)} samples using fitted parameters")
            
            # Extract features using fitted parameters
            features = self._extract_features_internal(data)
            X = features['X']
            y = features['y']
            
            # Apply fitted scaler if available
            if self.scaler is not None:
                X = self.scaler.transform(X)
                print(f"   Applied fitted {self.feature_config['normalization']} scaling")
        
        return {
            'X': X,
            'y': y,
            'feature_names': self.feature_names,
            'sample_ids': features.get('sample_ids', list(range(len(data))))
        }
    
    def fit_transform(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit parameters on training data and return transformed features.
        
        Args:
            train_data: Training data to fit parameters on and transform
            
        Returns:
            Dictionary containing transformed training features and labels
        """
        self.fit(train_data)
        return self.transform(train_data)
    
    def _extract_features_internal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Internal method to extract features using fitted or current parameters.
        
        Args:
            data: Data to extract features from
            
        Returns:
            Dictionary with raw features before scaling/imputation
        """
        with self._suppress_output():
            print(f"   Extracting features for {len(data)} samples...")
            
            # Initialize feature containers
            X_parts = []
            feature_names = []
            
            # 1. Extract Node Features (per-residue features)
            node_features_enabled = (self.feature_config['include_sequence_features'] or 
                                   self.feature_config['include_structure_features'] or
                                   self.feature_config['include_plm_embeddings'])
            
            if node_features_enabled:
                print("   Extracting node features...")
                node_extractor = NodeFeatures(data)
                node_features = node_extractor.profile_features(
                    include_sequence_features=self.feature_config['include_sequence_features'],
                    include_structure_features=self.feature_config['include_structure_features'],
                    include_plm_embeddings=self.feature_config['include_plm_embeddings'],
                    embedding_level=self.feature_config.get('embedding_level', 'residue')
                )
                
                # Process node features for each sample
                pooling_method = self.feature_config.get('pooling_method', 'mean')
                print(f"   Node Pooling method: {pooling_method}")
                decay_strength = self.feature_config.get('decay_strength', 1.0)
                
                for i in range(len(data)):
                    sample_node_parts = []
                    
                    # Process each feature type
                    for feat_type in ['sequence', 'structure', 'plm']:
                        if (feat_type in node_features and node_features[feat_type] is not None and
                            self.feature_config.get(f'include_{feat_type}_features', False) or
                            (feat_type == 'plm' and self.feature_config.get('include_plm_embeddings', False))):
                            
                            # Extract position-wise features for this sample
                            feat_df = node_features[feat_type]
                            if i < len(feat_df):
                                feat_matrix = self._extract_position_matrix(feat_df.iloc[i])
                                if feat_matrix is not None and feat_matrix.size > 0:
                                    # Apply pooling
                                    pooled_features = self._pool_positions(
                                        feat_matrix, pooling_method, decay_strength
                                    )
                                    if pooled_features is not None:
                                        sample_node_parts.append(pooled_features.astype(np.float32))
                    
                    # Concatenate node features for this sample
                    if sample_node_parts:
                        sample_node_vector = np.concatenate(sample_node_parts, axis=0)
                        X_parts.append(sample_node_vector)
                    else:
                        X_parts.append(np.array([]))
                
                # Generate node feature names
                if X_parts and len(X_parts[0]) > 0:
                    node_feature_count = len(X_parts[0])
                    feature_names.extend([f'node_{i}' for i in range(node_feature_count)])
                    print(f"   Node features: {node_feature_count} dimensions")
            
            # 2. Extract Global Features (per-window features)
            global_features_enabled = any([
                self.feature_config.get('include_weighted_metrix', False),
                self.feature_config.get('include_cksaap', False),
                self.feature_config.get('include_cpp', False),
                self.feature_config.get('include_coevolution_patterns', False)
            ])
            
            if global_features_enabled and self.fitted_global_features is not None:
                print("   Extracting global features...")
                
                # Get global features using fitted profiles
                global_feat_dict = self.fitted_global_features.profile_features(
                    data, 
                    return_fusion_format=True
                )
                
                # Process global features for each sample
                for i in range(len(data)):
                    sample_global_parts = []
                    
                    # Concatenate enabled global feature types
                    # Note: global_feat_dict contains numpy arrays when return_fusion_format=True
                    if self.feature_config.get('include_weighted_metrix', False) and 'weighted' in global_feat_dict:
                        weighted_array = global_feat_dict['weighted']
                        if i < len(weighted_array):
                            weighted_data = weighted_array[i]
                            if not isinstance(weighted_data, np.ndarray):
                                weighted_data = np.array(weighted_data)
                            sample_global_parts.append(weighted_data.flatten().astype(np.float32))
                    
                    if self.feature_config.get('include_cksaap', False) and 'cksaap' in global_feat_dict:
                        cksaap_array = global_feat_dict['cksaap']
                        if i < len(cksaap_array):
                            cksaap_data = cksaap_array[i]
                            if not isinstance(cksaap_data, np.ndarray):
                                cksaap_data = np.array(cksaap_data)
                            sample_global_parts.append(cksaap_data.flatten().astype(np.float32))
                    
                    if self.feature_config.get('include_cpp', False) and 'cpp' in global_feat_dict:
                        cpp_array = global_feat_dict['cpp']
                        if i < len(cpp_array):
                            cpp_data = cpp_array[i]
                            if not isinstance(cpp_data, np.ndarray):
                                cpp_data = np.array(cpp_data)
                            sample_global_parts.append(cpp_data.flatten().astype(np.float32))
                    
                    if self.feature_config.get('include_coevolution_patterns', False) and 'coevolution_patterns' in global_feat_dict:
                        coevo_array = global_feat_dict['coevolution_patterns']
                        if i < len(coevo_array):
                            coevo_data = coevo_array[i]
                            if not isinstance(coevo_data, np.ndarray):
                                coevo_data = np.array(coevo_data)
                            sample_global_parts.append(coevo_data.flatten().astype(np.float32))
                    
                    # Concatenate global features for this sample
                    if sample_global_parts:
                        sample_global_vector = np.concatenate(sample_global_parts, axis=0)
                        
                        # Combine with node features
                        if i < len(X_parts) and len(X_parts[i]) > 0:
                            X_parts[i] = np.concatenate([sample_global_vector, X_parts[i]], axis=0)
                        else:
                            X_parts.append(sample_global_vector)
                
                # Add global feature names
                global_feature_count = 0
                if self.feature_config.get('include_weighted_metrix', False):
                    global_feature_count += 14  # 14 weighted matrices
                if self.feature_config.get('include_cksaap', False):
                    global_feature_count += 1600  # 1600 CKSAAP features
                if self.feature_config.get('include_cpp', False):
                    global_feature_count += 100  # 100 CPP features
                if self.feature_config.get('include_coevolution_patterns', False):
                    global_feature_count += 50  # 50 coevolution features
                
                global_names = [f'global_{i}' for i in range(global_feature_count)]
                feature_names = global_names + feature_names
                print(f"   Global features: {global_feature_count} dimensions")
            
            # 3. Convert to final numpy array
            if X_parts and all(len(x) > 0 for x in X_parts):
                X = np.stack(X_parts, axis=0)
            elif X_parts:
                # Handle cases where some samples have no features
                max_dim = max(len(x) for x in X_parts if len(x) > 0)
                X_filled = []
                for x in X_parts:
                    if len(x) == 0:
                        X_filled.append(np.zeros(max_dim, dtype=np.float32))
                    else:
                        X_filled.append(x)
                X = np.stack(X_filled, axis=0)
            else:
                raise ValueError(" No features extracted")
            
            # 4. Extract labels 
            y = data['known_cleavage_site'].values if 'known_cleavage_site' in data.columns else np.zeros(len(data))
            
            print(f"   Final feature matrix: {X.shape}")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'sample_ids': list(range(len(data)))
        }
    
    def _extract_position_matrix(self, row_series: pd.Series) -> Optional[np.ndarray]:
        """
        Convert row series with position columns to matrix.
        
        Args:
            row_series: Pandas series with position-based columns (pos_0, pos_1, etc.)
            
        Returns:
            Matrix with shape (positions, features) or None if no valid data
        """
        if row_series is None:
            return None
        
        # Get position columns in order
        pos_cols = sorted([col for col in row_series.index if str(col).startswith('pos_')], 
                         key=lambda c: int(str(c).split('_')[-1]))
        
        if not pos_cols:
            return None
        
        vectors = []
        for col in pos_cols:
            value = row_series[col]
            if isinstance(value, list):
                vector = np.array(value)
            elif isinstance(value, np.ndarray):
                vector = value
            else:
                vector = np.array([value])  # Single value
            vectors.append(np.ravel(vector))
        
        if not vectors:
            return None
        
        try:
            return np.stack(vectors, axis=0)  # Shape: (positions, features)
        except ValueError:
            # Handle inconsistent vector sizes by padding with zeros
            max_len = max(len(v) for v in vectors)
            padded_vectors = []
            for v in vectors:
                if len(v) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(v)] = v
                    padded_vectors.append(padded)
                else:
                    padded_vectors.append(v)
            return np.stack(padded_vectors, axis=0)
    
    def _pool_positions(self, matrix: np.ndarray, method: str = 'mean', 
                       decay_strength: float = 1.0) -> Optional[np.ndarray]:
        """
        Pool features across positions using specified method.
        
        Args:
            matrix: Feature matrix with shape (positions, features)
            method: Pooling method ('mean', 'max', 'sum', 'linear_decay')
            decay_strength: Strength of decay for linear_decay method (0.0 to 1.0)
            
        Returns:
            Pooled feature vector or None if input is invalid
        """
        if matrix is None or matrix.size == 0:
            return None
        
        if len(matrix.shape) == 1:
            return matrix  # Already pooled
        
        L, D = matrix.shape
        
        if L == 1:
            return matrix[0]  # Single position
        
        if method == 'mean':
            return np.mean(matrix, axis=0)
        elif method == 'max':
            return np.max(matrix, axis=0)
        elif method == 'sum':
            return np.sum(matrix, axis=0)
        elif method == 'linear_decay':
            # Weight positions by distance from center
            center = L // 2
            distances = np.abs(np.arange(L) - center).astype(float)
            max_distance = float(max(center, L - 1 - center)) if L > 1 else 1.0
            
            if max_distance == 0:
                weights = np.ones(L, dtype=float)
            else:
                decay_strength = float(np.clip(decay_strength, 0.0, 1.0))
                weights = 1.0 - decay_strength * (distances / max_distance)
                weights = np.clip(weights, 0.0, None)
            
            # Normalize weights
            total_weight = weights.sum()
            if total_weight <= 0:
                weights = np.ones(L, dtype=float)
                total_weight = L
            
            weights = weights / total_weight
            
            # Apply weighted average
            return (weights[:, None] * matrix).sum(axis=0)
        else:
            # Default to mean for unknown methods
            return np.mean(matrix, axis=0)
    
