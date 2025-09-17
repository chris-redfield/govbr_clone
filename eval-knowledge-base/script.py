import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.io as pio

# Set Plotly's default renderer to 'browser' to prevent output to terminal
pio.renderers.default = 'browser'

def load_embeddings(file_path):
    """Loads embedding data from a CSV file."""
    df = pd.read_csv(file_path)
    embedding_cols = [col for col in df.columns if df[col].dtype == 'float64']
    embeddings = df[embedding_cols].values
    questions = df['pergunta'].tolist()
    return embeddings, questions

def apply_pca(embeddings_1, embeddings_2, n_components=2):
    """
    Applies PCA for dimensionality reduction on combined embeddings.
    Returns the reduced embeddings and the total explained variance ratio.
    """
    combined_embeddings = np.vstack([embeddings_1, embeddings_2])
    pca = PCA(n_components=n_components, random_state=42)
    combined_reduced = pca.fit_transform(combined_embeddings)
    
    n_samples_1 = len(embeddings_1)
    reduced_1 = combined_reduced[:n_samples_1]
    reduced_2 = combined_reduced[n_samples_1:]
    
    total_explained_variance = pca.explained_variance_ratio_.sum()
    return reduced_1, reduced_2, total_explained_variance

def calculate_distances_and_similarity(embeddings_kb, embeddings_chats):
    """
    Calculates the Euclidean distance and cosine similarity from each chat
    question to the nearest knowledge base question.
    """
    results = []
    for chat_emb in embeddings_chats:
        diff = embeddings_kb - chat_emb
        squared_distances = np.sum(diff**2, axis=1)
        min_distance = np.sqrt(np.min(squared_distances))
        
        similarities = cosine_similarity([chat_emb], embeddings_kb)[0]
        max_similarity = np.max(similarities)
        
        results.append((min_distance, max_similarity))
        
    distances, similarities = zip(*results)
    return list(distances), list(similarities)

def get_most_distant_questions(questions_chats, distances, similarities, n=10):
    """
    Identifies the n most distant chat questions and their distances.
    Also returns the indices of these questions.
    """
    results_df = pd.DataFrame({
        'pergunta_chat': questions_chats,
        'distancia_minima_euclidiana': distances,
        'similaridade_maxima_cosseno': similarities
    }).sort_values(by='distancia_minima_euclidiana', ascending=False)
    
    most_distant_indices = results_df.head(n).index.tolist()
    
    return results_df.head(n), most_distant_indices

def plot_2d_pca_with_highlights(reduced_kb, reduced_chats, questions_kb, questions_chats, most_distant_indices, title):
    """
    Creates a 2D visualization of the embeddings with the most distant
    chat questions highlighted in yellow.
    """
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        reduced_kb[:, 0], 
        reduced_kb[:, 1],
        label='embeddings_df', 
        alpha=0.7,
        s=50,
        color='blue'
    )
    
    plt.scatter(
        reduced_chats[:, 0], 
        reduced_chats[:, 1],
        label='embeddings_df_chats', 
        alpha=0.7,
        s=50,
        color='red'
    )
    
    most_distant_points = reduced_chats[most_distant_indices]
    plt.scatter(
        most_distant_points[:, 0],
        most_distant_points[:, 1],
        label='Top 10 Mais Distantes',
        alpha=1.0,
        s=100,
        color='yellow',
        edgecolor='black',
        linewidth=1.5
    )
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_3d_plot(reduced_kb, reduced_chats, questions_kb, questions_chats, most_distant_indices, title):
    """
    Create interactive 3D visualization using Plotly for both datasets,
    highlighting the most distant questions.
    """
    fig = go.Figure()

    hover_text_1 = [f"Dataset: embeddings_df<br>Pergunta: {pergunta}"
                    for pergunta in questions_kb]

    fig.add_trace(go.Scatter3d(
        x=reduced_kb[:, 0],
        y=reduced_kb[:, 1],
        z=reduced_kb[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.7
        ),
        name='embeddings_df',
        text=hover_text_1,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    hover_text_2 = [f"Dataset: embeddings_df_chats<br>Pergunta: {pergunta}"
                    for pergunta in questions_chats]

    fig.add_trace(go.Scatter3d(
        x=reduced_chats[:, 0],
        y=reduced_chats[:, 1],
        z=reduced_chats[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            opacity=0.7
        ),
        name='embeddings_df_chats',
        text=hover_text_2,
        hovertemplate='%{text}<extra></extra>'
    ))

    most_distant_points = reduced_chats[most_distant_indices]
    most_distant_questions_text = [questions_chats[i] for i in most_distant_indices]
    hover_text_3 = [f"Dataset: embeddings_df_chats (Most Distant)<br>Pergunta: {pergunta}"
                    for pergunta in most_distant_questions_text]
    
    fig.add_trace(go.Scatter3d(
        x=most_distant_points[:, 0],
        y=most_distant_points[:, 1],
        z=most_distant_points[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='yellow',
            opacity=1.0,
            symbol='circle',
            line=dict(color='black', width=1.5)
        ),
        name='Top 10 Most Distant',
        text=hover_text_3,
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1024,
        height=768,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()

if __name__ == "__main__":
    try:
        embeddings_kb, questions_kb = load_embeddings('perguntas_scripts_embeddings.csv')
        embeddings_chats, questions_chats = load_embeddings('perguntas_dos_chat_embeddings.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CSV files are in the same directory as the script.")
        exit()
    
    print("Calculating distances and similarities between real-world questions and the knowledge base...")
    distances, similarities = calculate_distances_and_similarity(embeddings_kb, embeddings_chats)
    
    most_distant_questions, most_distant_indices = get_most_distant_questions(questions_chats, distances, similarities)
    
    print("\nTop 10 most distant questions:")
    n = 1
    for idx, row in most_distant_questions.iterrows():
        print(f"{n}. ({idx}), Question: '{row.pergunta_chat}', Euclidean distance: {row.distancia_minima_euclidiana:.4f}, Cosine similarity: {row.similaridade_maxima_cosseno:.4f}")
        n += 1
        
    print("\nApplying PCA for 2D visualization...")
    reduced_kb_2d, reduced_chats_2d, total_explained_variance_2d = apply_pca(embeddings_kb, embeddings_chats, n_components=2)
    
    print(f"Total explained variance with 2-component PCA: {total_explained_variance_2d:.4f}")
    
    plot_2d_pca_with_highlights(reduced_kb_2d, reduced_chats_2d, questions_kb, questions_chats, most_distant_indices, "2D PCA Visualization with Most Distant Questions Highlighted")
    
    print("\nApplying PCA for 3D visualization...")
    reduced_kb_3d, reduced_chats_3d, total_explained_variance_3d = apply_pca(embeddings_kb, embeddings_chats, n_components=3)
    
    print(f"Total explained variance with 3-component PCA: {total_explained_variance_3d:.4f}")

    create_3d_plot(reduced_kb_3d, reduced_chats_3d, questions_kb, questions_chats, most_distant_indices, "3D PCA Visualization with Most Distant Questions Highlighted")