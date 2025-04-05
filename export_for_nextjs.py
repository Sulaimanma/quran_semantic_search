import os
import logging
from tqdm import tqdm
import numpy as np
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import umap
from sklearn.cluster import KMeans
from openai import OpenAI

client = OpenAI()

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
    load_dotenv()
    logger.info("Environment variables loaded")
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
    if not qdrant_client.collection_exists(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        exit(1)
    collection_info = qdrant_client.get_collection(collection_name)
    points_count = collection_info.points_count
    logger.info(f"Collection has {points_count} points")
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

def llm_summarize_cluster(text_list, max_words=3):
    """
    Use OpenAI's GPT-3.5-turbo to summarize the combined text of the cluster 
    in at most max_words that capture its core meaning.
    """
    max_texts = 50  # Limit texts to avoid token limit
    if len(text_list) > max_texts:
        import random
        random.seed(42)  # For reproducibility
        sampled_texts = random.sample(text_list, max_texts)
    else:
        sampled_texts = text_list
    combined_text = " ".join(sampled_texts)
    max_chars = 12000  # Conservative character limit
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "..."
    prompt = (
        f"Summarize the following text in at most {max_words} words to capture its core meaning:\n\n"
        f"{combined_text}\n\nSummary:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        summary_words = summary.split()
        if len(summary_words) > max_words:
            summary = " ".join(summary_words[:max_words])
        return summary
    except Exception as e:
        logging.error(f"Error in LLM summarization: {str(e)}")
        return ""

def reduce_dimensions_and_cluster(points, logger, n_clusters=20):
    """
    Reduce the high-dimensional embeddings to 3D using UMAP,
    cluster them with K-means, and compute a concise (≤3 words) core meaning
    for each cluster using an LLM summarizer.
    """
    logger.info("Reducing dimensions with UMAP")
    vectors = [p.vector for p in points]
    metadata = [{
        'id': p.id,
        'surah_id': p.payload.get('surah_id'),
        'surah_name': p.payload.get('surah_name'),
        'ayah': p.payload.get('ayah'),
        'arabic_text': p.payload.get('arabic_text'),
        'english_translation': p.payload.get('english_translation'),
    } for p in points]
    vectors_array = np.array(vectors)
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(vectors_array)
    logger.info(f"Clustering data into {n_clusters} semantic groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors_array)
    centroids = kmeans.cluster_centers_

    # Group English translations by cluster
    cluster_texts = {}
    for i, meta in enumerate(metadata):
        cid = clusters[i]
        cluster_texts.setdefault(cid, []).append(meta["english_translation"])

    # Create cluster summaries using the LLM summarizer
    cluster_summaries = {}
    for cluster_id in range(n_clusters):
        texts = cluster_texts.get(cluster_id, [])
        core_meaning = llm_summarize_cluster(texts, max_words=3)
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        if not indices:
            continue
        cluster_vectors = vectors_array[indices]
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        rep_index = indices[np.argmin(distances)]
        rep_meta = metadata[rep_index]
        cluster_summaries[str(cluster_id)] = {
            "representative_id": rep_meta["id"],
            "surah_id": rep_meta["surah_id"],
            "surah_name": rep_meta["surah_name"],
            "ayah": rep_meta["ayah"],
            "core_meaning": core_meaning,  # Core summary from LLM
            "arabic_text": rep_meta["arabic_text"],
            "centroid": centroid.tolist()
        }

    # Combine points data with 3D coordinates and cluster assignments
    points_data = []
    for i, meta in enumerate(metadata):
        point = {
            **meta,
            "x": float(embedding_3d[i, 0]),
            "y": float(embedding_3d[i, 1]),
            "z": float(embedding_3d[i, 2]),
            "cluster": int(clusters[i])
        }
        points_data.append(point)
    
    # Update each point with the cluster core meaning
    for point in points_data:
        cluster_id = str(point["cluster"])
        point["core_meaning"] = cluster_summaries.get(cluster_id, {}).get("core_meaning", "")

    return {
        "points": points_data,
        "clusters": cluster_summaries
    }

def export_json(data, output_file, logger):
    """Export data to JSON file for Next.js"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    base_name, extension = os.path.splitext(output_file)
    counter = 1
    current_file = output_file
    while os.path.exists(current_file):
        current_file = f"{base_name}_{counter}{extension}"
        counter += 1
    logger.info(f"Exporting data to {current_file}")
    with open(current_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Data successfully exported to {current_file}")

def main():
    logger = setup_logging()
    logger.info("Starting Quran embedding export for Next.js")
    qdrant_client = initialize_qdrant_client(logger)
    collection_name = "quran_verses"
    points = fetch_embeddings(qdrant_client, collection_name, logger)
    data = reduce_dimensions_and_cluster(points, logger)
    export_json(data, "public/quran_embeddings.json", logger)
    logger.info("Export process complete!")

if __name__ == "__main__":
    main()
