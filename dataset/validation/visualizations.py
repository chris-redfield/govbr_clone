import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def create_3d_cluster_visualization(df_clustered, tfidf_matrix, method='tsne', sample_size=5000, use_embeddings=False):
    """
    Create 3D visualization of clusters using dimensionality reduction
    
    Parameters:
    df_clustered: DataFrame with cluster labels
    tfidf_matrix: TF-IDF matrix from vectorization
    method: 'tsne', 'pca', or 'both'
    sample_size: number of points to sample for faster visualization
    """
    
    # Sample data if dataset is too large for performance
    if len(df_clustered) > sample_size:
        print(f"Sampling {sample_size} points from {len(df_clustered)} total points for visualization...")
        sample_idx = np.random.choice(len(df_clustered), sample_size, replace=False)
        df_sample = df_clustered.iloc[sample_idx].copy()
        tfidf_sample = tfidf_matrix[sample_idx]
    else:
        df_sample = df_clustered.copy()
        tfidf_sample = tfidf_matrix
    
    # Define colors for clusters
    colors = px.colors.qualitative.Set3[:len(df_sample['cluster'].unique())]
    
    # Method 1: t-SNE 3D
    if method in ['tsne', 'both']:
        print("Computing t-SNE 3D embedding...")
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
        if use_embeddings:
            tsne_coords = tsne_3d.fit_transform(tfidf_sample)
        else:
            tsne_coords = tsne_3d.fit_transform(tfidf_sample.toarray())
        # Remove outliers here
        # tsne_coords, _ = remove_tsne_outliers(tsne_coords)
        
        # Create interactive 3D plot with Plotly
        fig_tsne = go.Figure()
        
        for cluster_id in sorted(df_sample['cluster'].unique()):
            cluster_mask = df_sample['cluster'] == cluster_id
            cluster_data = df_sample[cluster_mask]
            cluster_coords = tsne_coords[cluster_mask]
            
            # Create hover text with message content
            hover_text = [f"Cluster: {cluster_id}<br>Author: {author}<br>Content: {content[:100]}..." 
                         for author, content in zip(cluster_data['name'], cluster_data['content'])]
            
            fig_tsne.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.7
                ),
                name=f'Cluster {cluster_id} ({len(cluster_data)} msgs)',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig_tsne.update_layout(
            title="3D t-SNE Visualization of Chat Clusters",
            scene=dict(
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                zaxis_title="t-SNE 3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1024,
            height=768
        )
        
        fig_tsne.show()
    
    # Method 2: PCA 3D
    if method in ['pca', 'both']:
        print("Computing PCA 3D embedding...")
        pca_3d = PCA(n_components=3, random_state=42)
        
        if use_embeddings:
            pca_coords = pca_3d.fit_transform(tfidf_sample)
        else:
            pca_coords = pca_3d.fit_transform(tfidf_sample.toarray())
        
        
        # Create interactive 3D plot with Plotly
        fig_pca = go.Figure()
        
        for cluster_id in sorted(df_sample['cluster'].unique()):
            cluster_mask = df_sample['cluster'] == cluster_id
            cluster_data = df_sample[cluster_mask]
            cluster_coords = pca_coords[cluster_mask]
            
            # Create hover text with message content
            hover_text = [f"Cluster: {cluster_id}<br>Author: {author}<br>Content: {content[:100]}..." 
                         for author, content in zip(cluster_data['name'], cluster_data['content'])]
            
            fig_pca.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.7
                ),
                name=f'Cluster {cluster_id} ({len(cluster_data)} msgs)',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig_pca.update_layout(
            title=f"3D PCA Visualization of Chat Clusters<br>Explained Variance: {pca_3d.explained_variance_ratio_.sum():.1%}",
            scene=dict(
                xaxis_title=f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700
        )
        
        fig_pca.show()
    
    return tsne_coords if method in ['tsne', 'both'] else None, pca_coords if method in ['pca', 'both'] else None

def create_2d_comparison_plots(df_clustered, tfidf_matrix, sample_size=5000, use_embeddings=False):
    """Create 2D comparison plots for better understanding"""
    
    # Sample data if needed
    if len(df_clustered) > sample_size:
        sample_idx = np.random.choice(len(df_clustered), sample_size, replace=False)
        df_sample = df_clustered.iloc[sample_idx].copy()
        tfidf_sample = tfidf_matrix[sample_idx]
    else:
        df_sample = df_clustered.copy()
        tfidf_sample = tfidf_matrix
    
    print("Computing 2D embeddings for comparison...")
    
    # Compute both t-SNE and PCA in 2D
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)

    if use_embeddings:
        tsne_coords_2d = tsne_2d.fit_transform(tfidf_sample)
    else:
        tsne_coords_2d = tsne_2d.fit_transform(tfidf_sample.toarray())

    
    pca_2d = PCA(n_components=2, random_state=42)

    if use_embeddings:
        pca_coords_2d = pca_2d.fit_transform(tfidf_sample)
    else:
        pca_coords_2d = pca_2d.fit_transform(tfidf_sample.toarray())
    
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # t-SNE 2D plot
    scatter1 = ax1.scatter(tsne_coords_2d[:, 0], tsne_coords_2d[:, 1], 
                          c=df_sample['cluster'], cmap='tab10', alpha=0.6, s=20)
    ax1.set_title('2D t-SNE Visualization', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.grid(True, alpha=0.3)
    
    # PCA 2D plot
    scatter2 = ax2.scatter(pca_coords_2d[:, 0], pca_coords_2d[:, 1], 
                          c=df_sample['cluster'], cmap='tab10', alpha=0.6, s=20)
    ax2.set_title(f'2D PCA Visualization\n(Explained Variance: {pca_2d.explained_variance_ratio_.sum():.1%})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.show()
    
    return tsne_coords_2d, pca_coords_2d

def analyze_cluster_separation(df_clustered, coords_3d, method_name):
    """Analyze how well separated the clusters are"""
    from scipy.spatial.distance import pdist, squareform
    
    print(f"\n{method_name} Cluster Separation Analysis:")
    print("="*50)
    
    # Calculate centroid distances
    centroids = {}
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_mask = df_clustered['cluster'] == cluster_id
        cluster_coords = coords_3d[cluster_mask]
        centroids[cluster_id] = np.mean(cluster_coords, axis=0)
    
    # Calculate distances between centroids
    centroid_coords = np.array(list(centroids.values()))
    distances = pdist(centroid_coords)
    distance_matrix = squareform(distances)
    
    print(f"Average distance between cluster centroids: {np.mean(distances):.3f}")
    print(f"Min distance between centroids: {np.min(distances):.3f}")
    print(f"Max distance between centroids: {np.max(distances):.3f}")
    
    # Find closest and farthest cluster pairs
    min_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(centroids)) * 1e6), distance_matrix.shape)
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    
    print(f"Closest clusters: {min_idx[0]} and {min_idx[1]} (distance: {distance_matrix[min_idx]:.3f})")
    print(f"Farthest clusters: {max_idx[0]} and {max_idx[1]} (distance: {distance_matrix[max_idx]:.3f})")


def remove_tsne_outliers(coords, method='iqr', factor=1.5):
    """
    Remove outliers from t-SNE coordinates using various methods
    
    Parameters:
    coords: numpy array of coordinates (n_samples, n_dimensions)
    method: 'iqr', 'zscore', or 'isolation'
    factor: threshold factor for outlier detection
    
    Returns:
    coords_clean: coordinates with outliers removed
    mask: boolean mask indicating which points are kept
    """
    
    if method == 'iqr':
        # IQR method - remove points outside Q1 - factor*IQR to Q3 + factor*IQR
        print("Using IQR method for outlier detection...")
        mask = np.ones(len(coords), dtype=bool)
        
        for dim in range(coords.shape[1]):
            q1, q3 = np.percentile(coords[:, dim], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            dim_mask = (coords[:, dim] >= lower_bound) & (coords[:, dim] <= upper_bound)
            mask = mask & dim_mask
            
            outliers_in_dim = (~dim_mask).sum()
            print(f"   Dimension {dim}: removed {outliers_in_dim} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
    
    elif method == 'zscore':
        # Z-score method - remove points with |z-score| > factor
        print("Using Z-score method for outlier detection...")
        from scipy import stats
        
        mask = np.ones(len(coords), dtype=bool)
        
        for dim in range(coords.shape[1]):
            z_scores = np.abs(stats.zscore(coords[:, dim]))
            dim_mask = z_scores < factor
            mask = mask & dim_mask
            
            outliers_in_dim = (~dim_mask).sum()
            print(f"   Dimension {dim}: removed {outliers_in_dim} outliers (|z-score| > {factor})")
    
    elif method == 'isolation':
        # Isolation Forest method
        print("Using Isolation Forest for outlier detection...")
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(coords)
        mask = outlier_labels == 1  # 1 = inlier, -1 = outlier
        
        outliers_total = (outlier_labels == -1).sum()
        print(f"   Isolation Forest removed {outliers_total} outliers")
    
    elif method == 'percentile':
        # Simple percentile-based removal
        print(f"Using percentile method (keeping central {100-2*factor:.1f}%)...")
        mask = np.ones(len(coords), dtype=bool)
        
        for dim in range(coords.shape[1]):
            lower_percentile = factor
            upper_percentile = 100 - factor
            lower_bound, upper_bound = np.percentile(coords[:, dim], [lower_percentile, upper_percentile])
            
            dim_mask = (coords[:, dim] >= lower_bound) & (coords[:, dim] <= upper_bound)
            mask = mask & dim_mask
            
            outliers_in_dim = (~dim_mask).sum()
            print(f"   Dimension {dim}: removed {outliers_in_dim} outliers (keeping {lower_percentile}%-{upper_percentile}% range)")
    
    else:
        raise ValueError("Method must be one of: 'iqr', 'zscore', 'isolation', 'percentile'")
    
    coords_clean = coords[mask]
    
    print(f"Total outliers removed: {(~mask).sum()}/{len(coords)} ({(~mask).sum()/len(coords)*100:.1f}%)")
    print(f"Remaining points: {mask.sum()}")
    
    return coords_clean, mask