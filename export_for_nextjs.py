import os
import logging
from tqdm import tqdm
import numpy as np
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import umap
from sklearn.cluster import KMeans
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

def reduce_dimensions_and_cluster(points, logger, n_clusters=20):
    """Reduce dimensions and cluster the data based on semantic similarity"""
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
    
    # Apply K-means clustering to original embeddings for semantic grouping
    logger.info(f"Clustering data into {n_clusters} semantic groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors_array)
    
    logger.info("Processing complete")
    
    # Create result dictionary with both 3D coordinates and metadata
    result = []
    for i, meta in enumerate(metadata):
        result.append({
            **meta,
            'x': float(embedding_3d[i, 0]),
            'y': float(embedding_3d[i, 1]),
            'z': float(embedding_3d[i, 2]),
            'cluster': int(clusters[i])
        })
    
    return result

def export_json(data, output_file, logger):
    """Export data to JSON file for Next.js"""
    logger.info(f"Exporting data to {output_file}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Data successfully exported to {output_file}")
def main():
    logger = setup_logging()
    logger.info("Starting Quran embedding export for Next.js")
    
    # Initialize Qdrant client
    qdrant_client = initialize_qdrant_client(logger)
    
    # Define collection name
    collection_name = "quran_verses"
    
    # Fetch embeddings
    points = fetch_embeddings(qdrant_client, collection_name, logger)
    
    # Reduce dimensions and cluster
    data = reduce_dimensions_and_cluster(points, logger)
    
    # Export to JSON
    export_json(data, "public/quran_embeddings.json", logger)
    
    logger.info("Export process complete!")

if __name__ == "__main__":
    main()
