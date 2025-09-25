'''For PCA, t-SNE, UMAP plots'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import List
from sklearn.impute import SimpleImputer
import umap

class plot:
    def __init__(self):
        pass
    
    def plot_pca(self, labels: list, df: pd.DataFrame,
                             n_components: int = 2,
                             impute: bool = False,
                             standardize: bool = True,
                             figsize: tuple = (10, 8),
                             alpha: float = 0.7,
                             title: str = "PCA Analysis") -> tuple:
        """
        Create the PCA plot of features
        
        Parameters:
        -----------
        labels : list
            sample labels (0: non-cleavage site, 1: cleavage site)
        df : pd.DataFrame
            feature matrix, each row is a sample, each column is a feature
        n_components : int, default=2
            the number of PCA components
        impute : bool, default=False
            if True, impute the missing values by the mean of the column
        standardize : bool, default=True
            if True, standardize the features
        figsize : tuple, default=(10, 8)
            the size of the figure
        alpha : float, default=0.7
            the transparency of the points
        title : str
            the title of the plot
            
        Returns:
        --------
        tuple: (fig, ax, pca_object, transformed_data)
        """
        
        # Prepare the data
        X = df.values
        y = np.array(labels)
        
        print(f"The shape of the feature matrix: {X.shape}")
        print(f"The number of labels: {len(y)}")
        print(f"The number of cleavage site samples: {np.sum(y == 1)}")
        print(f"The number of non-cleavage site samples: {np.sum(y == 0)}")
        
        # Impute the missing values
        if impute:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            print("The missing values have been imputed")
        else:
            X_imputed = X

        # Standardize the features (recommended for PCA)
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            print("The features have been standardized")
        else:
            X_scaled = X_imputed
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Print the explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance ratio: {explained_variance}")
        print(f"The cumulative explained variance of the first {n_components} components: {np.sum(explained_variance):.3f}")
        
        # Create the figure with enhanced styling
        plt.rcParams.update({'font.size': 12})  # Increase base font size
        fig, ax = plt.subplots(figsize=figsize)

        # Define enhanced colors and labels (consistent with t-SNE plots)
        colors = ['#3498db', '#e74c3c']  # Blue and red with better contrast
        class_labels = ['Non-cleavage Window (nonCW)', 'Cleavage Window (CW)']

        # Plot the scatter plot with enhanced styling
        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = (y == i)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=color, label=f'{label} (n={np.sum(mask):,})',
                      alpha=alpha, s=60, edgecolors='white', linewidth=1.0)

        # Enhanced axis labels and title
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} explained variance)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} explained variance)', fontsize=16, fontweight='bold')
        ax.set_title(f'{title}', fontsize=20, fontweight='bold', pad=25)

        # Enhanced legend with consistent styling
        legend = ax.legend(loc='upper right', fontsize=14, frameon=True,
                          fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.5)

        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Enhanced grid and styling
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)

        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=14,
                      labelcolor='black', width=1.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Add subtle border styling
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        plt.show()
        
        return (fig, ax, pca, X_pca)
    
    def plot_tsne(self, labels: list, df: pd.DataFrame,
                  n_components: int = 2,
                  impute: bool = False,
                  standardize: bool = True,
                  figsize: tuple = (10, 8),
                  alpha: float = 0.7,
                  title: str = "t-SNE Analysis",
                  perplexity: float = 30.0,
                  max_iter: int = 1000,
                  random_state: int = 42) -> tuple:
        """
        Create the t-SNE plot of features
        
        Parameters:
        -----------
        labels : list
            sample labels (0: non-cleavage site, 1: cleavage site)
        df : pd.DataFrame
            feature matrix, each row is a sample, each column is a feature
        n_components : int, default=2
            the number of t-SNE components
        impute : bool, default=False
            if True, impute the missing values by the mean of the column
        standardize : bool, default=True
            if True, standardize the features
        figsize : tuple, default=(10, 8)
            the size of the figure
        alpha : float, default=0.7
            the transparency of the points
        title : str
            the title of the plot
        perplexity : float, default=30.0
            t-SNE perplexity parameter (should be between 5-50)
        max_iter : int, default=1000
            maximum number of iterations
        random_state : int, default=42
            random state for reproducibility
            
        Returns:
        --------
        tuple: (fig, ax, tsne_object, transformed_data)
        """
        
        # Prepare the data
        X = df.values
        y = np.array(labels)
        
        print(f"The shape of the feature matrix: {X.shape}")
        print(f"The number of labels: {len(y)}")
        print(f"The number of cleavage site samples: {np.sum(y == 1)}")
        print(f"The number of non-cleavage site samples: {np.sum(y == 0)}")
        
        # Adjust perplexity if necessary
        max_perplexity = (X.shape[0] - 1) / 3
        if perplexity > max_perplexity:
            perplexity = max(5, int(max_perplexity))
            print(f"Perplexity adjusted to {perplexity} due to small sample size")
        
        # Impute the missing values
        if impute:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            print("The missing values have been imputed")
        else:
            X_imputed = X

        # Standardize the features (recommended for t-SNE)
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            print("The features have been standardized")
        else:
            X_scaled = X_imputed
        
        # Perform t-SNE
        print(f"Running t-SNE with perplexity={perplexity}, max_iter={max_iter}...")
        tsne = TSNE(n_components=n_components, 
                   perplexity=perplexity,
                   max_iter=max_iter,
                   random_state=random_state,
                   verbose=1)
        X_tsne = tsne.fit_transform(X_scaled)
        
        print(f"t-SNE completed, KL divergence: {tsne.kl_divergence_:.3f}")
        
        # Create the figure with enhanced styling
        plt.rcParams.update({'font.size': 12})  # Increase base font size
        fig, ax = plt.subplots(figsize=figsize)

        # Define enhanced colors and labels (consistent with RSA/AAindex plots)
        colors = ['#3498db', '#e74c3c']  # Blue and red with better contrast
        class_labels = ['Non-cleavage Window (nonCW)', 'Cleavage Window (CW)']

        # Plot the scatter plot with enhanced styling
        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = (y == i)
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=color, label=f'{label} (n={np.sum(mask):,})',
                      alpha=alpha, s=60, edgecolors='white', linewidth=1.0)

        # Enhanced axis labels and title
        ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
        ax.set_title(f'{title}', fontsize=20, fontweight='bold', pad=25)

        # Enhanced legend with consistent styling
        legend = ax.legend(loc='upper right', fontsize=14, frameon=True,
                          fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.5)

        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Enhanced grid and styling
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)

        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=14,
                      labelcolor='black', width=1.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Add subtle border styling
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.5)

        # Add technical info in bottom corner (smaller, less prominent)
        tech_info = f'Perplexity: {perplexity} | KL Divergence: {tsne.kl_divergence_:.3f}'
        ax.text(0.02, 0.02, tech_info, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
                fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        return (fig, ax, tsne, X_tsne)
    
    def plot_umap(self, labels: list, df: pd.DataFrame,
                  n_components: int = 2,
                  impute: bool = False,
                  standardize: bool = True,
                  figsize: tuple = (10, 8),
                  alpha: float = 0.7,
                  title: str = "UMAP Analysis",
                  n_neighbors: int = 15,
                  min_dist: float = 0.1,
                  metric: str = 'euclidean',
                  random_state: int = 42) -> tuple:
        """
        Create the UMAP plot of features
        
        Parameters:
        -----------
        labels : list
            sample labels (0: non-cleavage site, 1: cleavage site)
        df : pd.DataFrame
            feature matrix, each row is a sample, each column is a feature
        n_components : int, default=2
            the number of UMAP components
        impute : bool, default=False
            if True, impute the missing values by the mean of the column
        standardize : bool, default=True
            if True, standardize the features
        figsize : tuple, default=(10, 8)
            the size of the figure
        alpha : float, default=0.7
            the transparency of the points
        title : str
            the title of the plot
        n_neighbors : int, default=15
            the number of neighbors to consider for manifold approximation
        min_dist : float, default=0.1
            minimum distance between points in the low-dimensional representation
        metric : str, default='euclidean'
            the metric to use for distance computation
        random_state : int, default=42
            random state for reproducibility
            
        Returns:
        --------
        tuple: (fig, ax, umap_object, transformed_data)
        """
        
        
        # Prepare the data
        X = df.values
        y = np.array(labels)
        
        print(f"The shape of the feature matrix: {X.shape}")
        print(f"The number of labels: {len(y)}")
        print(f"The number of cleavage site samples: {np.sum(y == 1)}")
        print(f"The number of non-cleavage site samples: {np.sum(y == 0)}")
        
        # Adjust n_neighbors if necessary
        max_neighbors = X.shape[0] - 1
        if n_neighbors > max_neighbors:
            n_neighbors = max(2, max_neighbors)
            print(f"n_neighbors adjusted to {n_neighbors} due to small sample size")
        
        # Impute the missing values
        if impute:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            print("The missing values have been imputed")
        else:
            X_imputed = X

        # Standardize the features (recommended for UMAP)
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            print("The features have been standardized")
        else:
            X_scaled = X_imputed
        
        # Perform UMAP
        print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
        umap_reducer = umap.UMAP(n_components=n_components,
                               n_neighbors=n_neighbors,
                               min_dist=min_dist,
                               metric=metric,
                               random_state=random_state,
                               verbose=True)
        X_umap = umap_reducer.fit_transform(X_scaled)
        
        print("UMAP completed")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define the colors and labels
        colors = ['blue', 'red']
        class_labels = ['Non-cleavage Window (0)', 'Cleavage Window (1)']
        
        # Plot the scatter plot
        for i, (color, label) in enumerate(zip(colors, class_labels)):
            mask = (y == i)
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                      c=color, label=f'{label} (n={np.sum(mask)})', 
                      alpha=alpha, s=50, edgecolors='black', linewidth=0.5)
        
        # Set the labels and title
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} (neighbors={n_neighbors}, min_dist={min_dist})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return (fig, ax, umap_reducer, X_umap)
    
    def plot_comparison(self, labels: list, df: pd.DataFrame,
                       methods: List[str] = ['pca', 'tsne', 'umap'],
                       impute: bool = False,
                       standardize: bool = True,
                       figsize: tuple = (15, 5),
                       alpha: float = 0.7,
                       title: str = "Dimensionality Reduction Comparison") -> dict:
        """
        Compare the visualization effect of multiple dimensionality reduction methods
        
        Parameters:
        -----------
        labels : list
            sample labels (0: non-cleavage site, 1: cleavage site)
        df : pd.DataFrame
            feature matrix, each row is a sample, each column is a feature
        methods : List[str], default=['pca', 'tsne', 'umap']
            the list of dimensionality reduction methods to compare
        impute : bool, default=False
            if True, impute the missing values by the mean of the column
        standardize : bool, default=True
            if True, standardize the features
        figsize : tuple, default=(15, 5)
            the size of the figure
        alpha : float, default=0.7
            the transparency of the points
        title : str
            the title of the plot
            
        Returns:
        --------
        dict: a dictionary containing the results of each method
        """
        
        # Prepare the data
        X = df.values
        y = np.array(labels)
        
        # Impute the missing values
        if impute:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            print("The missing values have been imputed")
        else:
            X_imputed = X

        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # Create subplots
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        results = {}
        colors = ['blue', 'red']
        class_labels = ['Non-cleavage Window (0)', 'Cleavage Window (1)']
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            
            if method.lower() == 'pca':
                # PCA
                pca = PCA(n_components=2, random_state=42)
                X_transformed = pca.fit_transform(X_scaled)
                results['pca'] = (pca, X_transformed)
                
                # Plot
                for i, (color, label) in enumerate(zip(colors, class_labels)):
                    mask = (y == i)
                    ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                              c=color, label=f'{label} (n={np.sum(mask)})', 
                              alpha=alpha, s=30, edgecolors='black', linewidth=0.3)
                
                explained_var = pca.explained_variance_ratio_
                ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontweight='bold')
                ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontweight='bold')
                ax.set_title('PCA', fontweight='bold')
                
            elif method.lower() == 'tsne':
                # t-SNE
                perplexity = min(30, (X.shape[0] - 1) // 3)
                perplexity = max(5, perplexity)
                
                tsne = TSNE(n_components=2, perplexity=perplexity, 
                           random_state=42, n_iter=1000)
                X_transformed = tsne.fit_transform(X_scaled)
                results['tsne'] = (tsne, X_transformed)
                
                # Plot
                for i, (color, label) in enumerate(zip(colors, class_labels)):
                    mask = (y == i)
                    ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                              c=color, label=f'{label} (n={np.sum(mask)})', 
                              alpha=alpha, s=30, edgecolors='black', linewidth=0.3)
                
                ax.set_xlabel('t-SNE 1', fontweight='bold')
                ax.set_ylabel('t-SNE 2', fontweight='bold')
                ax.set_title(f't-SNE (perp={perplexity})', fontweight='bold')
                
            elif method.lower() == 'umap':
                
                n_neighbors = min(15, X.shape[0] - 1)
                n_neighbors = max(2, n_neighbors)
                
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                       random_state=42, verbose=False)
                X_transformed = umap_reducer.fit_transform(X_scaled)
                results['umap'] = (umap_reducer, X_transformed)
                
                # Plot
                for i, (color, label) in enumerate(zip(colors, class_labels)):
                    mask = (y == i)
                    ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                              c=color, label=f'{label} (n={np.sum(mask)})', 
                              alpha=alpha, s=30, edgecolors='black', linewidth=0.3)
                
                ax.set_xlabel('UMAP 1', fontweight='bold')
                ax.set_ylabel('UMAP 2', fontweight='bold')
                ax.set_title(f'UMAP (neighbors={n_neighbors})', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only show the legend in the first subplot
                ax.legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return results

    def plot_pca_explained_variance(self, df: pd.DataFrame,
                                  impute: bool = False,
                                  standardize: bool = True,
                                  figsize: tuple = (10, 6),
                                  title: str = "Explained Variance"):
        """
        Plot the PCA explained variance with dual y-axis (individual and cumulative)
        
        Parameters:
        -----------
        df : pd.DataFrame
            feature matrix
        impute : bool, default=False
            if True, impute the missing values by the mean of the column
        standardize : bool, default=True
            Whether to standardize features before PCA
        figsize : tuple, default=(10, 6)
            Figure size
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure: Generated figure
        """
        
        X = df.values

        # Impute the missing values
        if impute:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            print("The missing values have been imputed")
        else:
            X_imputed = X

        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed

        # Use all possible components (min of n_samples-1 and n_features)
        n_comp = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)
        
        # Extract variance information
        var_explained = pca.explained_variance_ratio_ * 100
        n_components = len(var_explained)
        components = np.arange(1, n_components + 1)
        cumulative = np.cumsum(var_explained)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot for individual variance
        bars = ax.bar(components, var_explained, 
                      alpha=0.75, color='#3498DB', 
                      edgecolor='black', linewidth=0.5,
                      label='Individual')
        
        # Add right axis for cumulative plot
        ax_twin = ax.twinx()
        
        # Line plot for cumulative variance
        line = ax_twin.plot(components, cumulative, 'ro-', 
                           linewidth=2, markersize=5,
                           label='Cumulative')
        
        # 90% reference line
        ax_twin.axhline(y=90, color='red', linestyle='--', alpha=0.5)
        
        # Configure variance plot with bold labels
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel('Principal Components (%)', fontweight='bold')
        ax.set_ylabel('Individual Variance (%)', fontweight='bold')
        
        # Set x-axis as percentage of total components
        ax.set_xticks(components)
        # Calculate the percentage labels for the x-axis
        if n_components > 10:
            # Only show the percentage of the first 10 components
            step = max(1, n_components // 10)
            ax.set_xticks(components[::step])
            labels = [f"{int(100*i/n_components)}%" for i in components[::step]]
            ax.set_xticklabels(labels, fontweight='bold')
        else:
            # For a small number of components, show all labels
            labels = [f"{int(100*i/n_components)}%" for i in components]
            ax.set_xticklabels(labels, fontweight='bold')
        
        ax.set_ylim([0, np.max(var_explained)*1.1])
        
        # Configure twin axis
        ax_twin.set_ylabel('Cumulative Variance (%)', fontweight='bold', color='red')
        ax_twin.spines['right'].set_color('red')
        ax_twin.tick_params(axis='y', colors='red', labelsize=10)
        ax_twin.set_ylim([0, 100])
        
        # Make all tick labels bold
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax_twin.get_yticklabels():
            label.set_fontweight('bold')
        
        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right',
                 frameon=True, framealpha=0.95, edgecolor='gray')
        
        plt.tight_layout()
        plt.show()