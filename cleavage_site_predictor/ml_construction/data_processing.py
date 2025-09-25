#!/usr/bin/env python3
"""
Data Processing Utilities for:

- Create balanced subdatasets
- CD-HIT clustering
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.utils import resample
import aaanalysis as aa
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split



def create_balanced_subdatasets(data: pd.DataFrame,
                                pos_neg_ratio: float = 1.0,
                                random_state: int = 42) -> List[pd.DataFrame]:
        """
        Create balanced subdatasets from imbalanced training data:
        - All positive samples are included in EVERY dataset
        - Negative samples are split WITHOUT overlap across datasets  
        - Each dataset requires at least half as many negatives as positives
        
        Example: 30 pos + 305 neg → 10 datasets:
            Dataset 1: 30 pos + neg[1-30]
            Dataset 2: 30 pos + neg[31-60] 
            Dataset 3: 30 pos + neg[61-90]
            ...
            Dataset 10: 30 pos + neg[271-300]
            Remaining 5 neg discarded (5 < 15, which is half of 30)
        
        Args:
            data: Training dataframe with 'known_cleavage_site' column
            pos_neg_ratio: Desired ratio of positive to negative samples (default: 1.0 for 1:1)
            min_segment_size: Minimum size for a valid segment (kept for compatibility)
            random_state: Random state for reproducibility
            
        Returns:
            List of balanced DataFrames with adequate pos-neg ratio
        """
        print(f"🔄 Creating balanced subdatasets (same pos in all, split neg across datasets)...")
        
        # Separate positive and negative samples
        positive_samples = data[data['known_cleavage_site'] == 1]#.reset_index(drop=True)
        negative_samples = data[data['known_cleavage_site'] == 0]#.reset_index(drop=True)
        
        n_pos = len(positive_samples)
        n_neg = len(negative_samples)
        
        print(f"   Available: {n_pos} positive, {n_neg} negative samples")
        
        if n_pos == 0:
            raise ValueError("No positive samples found in training data")
        
        if n_neg == 0:
            raise ValueError("No negative samples found in training data")
        
        # Shuffle negative samples for random distribution
        np.random.seed(random_state)
        negative_shuffled = negative_samples.sample(n=len(negative_samples), random_state=random_state)#.reset_index(drop=True)
        
        # For balanced datasets with pos_neg_ratio = 1.0:
        # Each dataset should have n_pos positive samples and n_pos negative samples
        if pos_neg_ratio == 1.0:
            neg_per_segment = n_pos  # Same number of neg as pos for balanced
        else:
            neg_per_segment = int(n_pos / pos_neg_ratio)
        
        # Calculate number of possible segments based on available negative samples
        n_segments = n_neg // neg_per_segment
        
        if n_segments == 0:
            print("    Not enough negative samples to create balanced segments")
            print(f"   Need {neg_per_segment} neg per segment, but only have {n_neg} total")
            # Fallback: create one segment with all available data
            return [data.copy()]
        
        # Ensure we have at least half the positives as negatives per segment
        min_neg_required = n_pos // 2  # At least half the positives
        if neg_per_segment < min_neg_required:
            print(f"    Calculated neg per segment ({neg_per_segment}) is below minimum ({min_neg_required})")
            # Adjust neg_per_segment to meet minimum requirement
            neg_per_segment = min_neg_required
            n_segments = n_neg // neg_per_segment
        
        print(f"   Creating {n_segments} balanced segments")
        print(f"   Each segment: {n_pos} pos + {neg_per_segment} neg = {n_pos + neg_per_segment} total")
        
        # Create balanced segments
        balanced_segments = []
        
        for segment_idx in range(n_segments):
            # Get ALL positive samples (same in every dataset)
            segment_pos = positive_samples.copy()
            
            # Get non-overlapping negative samples for this segment
            neg_start = segment_idx * neg_per_segment
            neg_end = neg_start + neg_per_segment
            segment_neg = negative_shuffled.iloc[neg_start:neg_end]
            
            # Combine positive and negative samples (preserve original indices!)
            segment_data = pd.concat([segment_pos, segment_neg], ignore_index=False)
            
            # Shuffle the combined segment
            segment_data = segment_data.sample(frac=1, random_state=random_state + segment_idx)#.reset_index(drop=False)
            
            balanced_segments.append(segment_data)
            
            print(f"     Segment {segment_idx + 1}: {len(segment_pos)} pos + {len(segment_neg)} neg = {len(segment_data)} total")
            print(f"       Neg samples: indices {neg_start}-{neg_end-1}")
        
        # Handle remaining negative samples
        remaining_neg_start = n_segments * neg_per_segment
        remaining_neg_count = n_neg - remaining_neg_start
        
        if remaining_neg_count > 0:
            print(f"   Remaining {remaining_neg_count} negative samples not used in segments")
            
            # Create an additional segment only if we have enough negatives (at least half the number of positives)
            min_neg_required = n_pos // 2  # At least half the positives
            
            if remaining_neg_count >= min_neg_required:
                remaining_neg = negative_shuffled.iloc[remaining_neg_start:]
                segment_pos = positive_samples.copy()
                
                # Create final segment with remaining negatives (preserve original indices!)
                segment_data = pd.concat([segment_pos, remaining_neg], ignore_index=False)
                segment_data = segment_data.sample(frac=1, random_state=random_state + n_segments)#.reset_index(drop=True)
                
                balanced_segments.append(segment_data)
                print(f"     Final Segment {len(balanced_segments)}: {len(segment_pos)} pos + {len(remaining_neg)} neg = {len(segment_data)} total")
            else:
                print(f"    Discarding remaining negatives: {remaining_neg_count} < {min_neg_required} (min required: half of {n_pos} positives)")
        
        print(f"    Created {len(balanced_segments)} balanced segments")
        print(f"   Key: Same {n_pos} positive samples in ALL segments, different negatives in each")
        
        return balanced_segments

