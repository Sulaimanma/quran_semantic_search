import os
import logging
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import umap
import pandas as pd

def setup_logging():
    """Configure and return a logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def initialize_qdrant_client(logger):
    """Initialize Qdrant client"""
    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("Qdrant client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        exit(1)
    
    return qdrant_client

def fetch_embeddings(qdrant_client, collection_name, logger):
    """Fetch all embeddings and metadata from Qdrant"""
    logger.info(f"Fetching embeddings from collection: {collection_name}")
    
    # Check if collection exists
    if not qdrant_client.collection_exists(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        exit(1)
    
    # Get collection info to determine size
    collection_info = qdrant_client.get_collection(collection_name)
    points_count = collection_info.points_count
    logger.info(f"Collection has {points_count} points")
    
    # Fetch all points with vectors (will paginate if needed)
    all_points = []
    limit = 1000  # Batch size
    offset = 0
    
    with tqdm(total=points_count, desc="Fetching points") as pbar:
        while offset < points_count:
            points = qdrant_client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )[0]
            
            if not points:
                break
                
            all_points.extend(points)
            offset += len(points)
            pbar.update(len(points))
    
    logger.info(f"Successfully fetched {len(all_points)} points")
    return all_points

def reduce_dimensions(points, logger):
    """Reduce embedding dimensions from 1536 to 3D using UMAP"""
    logger.info("Reducing dimensions with UMAP")
    
    # Extract vectors and metadata
    vectors = []
    metadata = []
    
    for point in points:
        vectors.append(point.vector)
        metadata.append({
            'id': point.id,
            'surah_id': point.payload.get('surah_id'),
            'surah_name': point.payload.get('surah_name'),
            'ayah': point.payload.get('ayah'),
            'arabic_text': point.payload.get('arabic_text'),
            'english_translation': point.payload.get('english_translation'),
        })
    
    # Convert to numpy array
    vectors_array = np.array(vectors)
    
    # Apply UMAP
    logger.info("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(vectors_array)
    
    logger.info("Dimension reduction complete")
    
    # Create a DataFrame with the 3D coordinates and metadata
    df = pd.DataFrame(metadata)
    df['x'] = embedding_3d[:, 0]
    df['y'] = embedding_3d[:, 1]
    df['z'] = embedding_3d[:, 2]
    
    return df

def create_visualization(df, output_file, logger):
    """Create 3D interactive visualization with Plotly"""
    logger.info("Creating 3D visualization")
    
    # Create hover text
    df['hover_text'] = df.apply(
        lambda row: f"Surah {row['surah_id']}: {row['surah_name']}<br>" +
                   f"Ayah {row['ayah']}<br>" +
                   f"<b>Arabic:</b> {row['arabic_text']}<br>" +
                   f"<b>English:</b> {row['english_translation']}",
        axis=1
    )
    
    # Create custom colormap based on Surah ID
    surah_ids = df['surah_id'].astype(int)
    max_surah_id = surah_ids.max()
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=4,
            color=surah_ids,
            colorscale='Viridis',
            colorbar=dict(title="Surah ID"),
            opacity=0.8
        ),
        text=df['hover_text'],
        hoverinfo='text'
    )])
    
    # Update layout
    fig.update_layout(
        title="Quran Verses in 3D Semantic Space",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        width=1200,
        height=800,
        margin=dict(r=0, l=0, b=0, t=40)
    )
    
    # Save the visualization to HTML
    fig.write_html(output_file)
    logger.info(f"Visualization saved to {output_file}")
    
    return fig

def main():
    logger = setup_logging()
    logger.info("Starting Quran embedding visualization")
    
    # Initialize Qdrant client
    qdrant_client = initialize_qdrant_client(logger)
    
    # Define collection name
    collection_name = "quran_verses"
    
    # Fetch embeddings
    points = fetch_embeddings(qdrant_client, collection_name, logger)
    
    # Reduce dimensions for visualization
    df = reduce_dimensions(points, logger)
    
    # Create and save visualization
    output_file = "quran_embeddings_visualization.html"
    fig = create_visualization(df, output_file, logger)
    
    logger.info("Visualization process complete!")
    logger.info(f"Open {output_file} in a web browser to explore the visualization")

if __name__ == "__main__":
    main()
