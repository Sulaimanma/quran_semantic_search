import json, os
import logging
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

def setup_logging():
    """Configure and return a logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def initialize_clients(logger):
    """Initialize OpenAI and Qdrant clients"""
    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        exit(1)
    logger.info("API key retrieved successfully")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("Qdrant client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        exit(1)
    
    return openai_client, qdrant_client

def setup_collection(qdrant_client, collection_name, logger):
    """Create collection if it doesn't exist"""
    logger.info(f"Using collection: {collection_name}")
    
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' does not exist. Creating...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created successfully")
    else:
        logger.info(f"Collection '{collection_name}' already exists")
    
    return collection_name

def get_embedding(text, client, logger, model="text-embedding-ada-002"):
    """Generate an embedding for the given text"""
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

def load_quran_data(file_path, logger):
    """Load Quran data from JSON file"""
    logger.info(f"Loading Quran data from {file_path}...")
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded Quran data with {len(data)} surahs")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        exit(1)

def generate_embeddings(data, openai_client, logger):
    """Generate embeddings for all verses in the Quran data"""
    points = []
    total_verses = sum(len(surah["verses"]) for surah in data)
    logger.info(f"Total verses to process: {total_verses}")
    
    # Initialize counters
    processed_surahs = 0
    processed_verses = 0
    embedding_errors = 0
    
    # Loop over each Surah and verse
    for surah in data:
        surah_id = surah["id"]
        surah_name = surah["name"]
        logger.info(f"Processing Surah {surah_id}: {surah_name} ({len(surah['verses'])} verses)")
        
        for verse in tqdm(surah["verses"], desc=f"Surah {surah_id}", leave=False):
            ayah = verse["id"]
            arabic_text = verse["text"]
            english_translation = verse["translation"]
            
            # Generate embedding
            try:
                logger.debug(f"Generating embedding for Surah {surah_id}, Ayah {ayah}")
                embedding = get_embedding(english_translation, openai_client, logger)
                
                # Create point with UUID
                original_id = f"{surah_id}:{ayah}:EN"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, original_id))
                metadata = {
                    "surah_id": surah_id,
                    "surah_name": surah_name,
                    "ayah": ayah,
                    "arabic_text": arabic_text,
                    "english_translation": english_translation,
                    "language": "EN",
                    "original_id": original_id  # Store the original ID in metadata
                }
                
                points.append(PointStruct(id=point_id, vector=embedding, payload=metadata))
                processed_verses += 1
                
            except Exception as e:
                logger.error(f"Error processing Surah {surah_id}, Ayah {ayah}: {e}")
                embedding_errors += 1
                continue
        
        processed_surahs += 1
        logger.info(f"Completed Surah {surah_id}: {processed_verses}/{total_verses} verses processed so far")
    
    return points, {
        "processed_surahs": processed_surahs,
        "total_surahs": len(data),
        "processed_verses": processed_verses,
        "total_verses": total_verses,
        "embedding_errors": embedding_errors
    }
def upload_to_qdrant(points, qdrant_client, collection_name, batch_size, logger):
    """Upload points to Qdrant in batches"""
    logger.info(f"Uploading {len(points)} points to Qdrant in batches of {batch_size}")
    
    successful_uploads = 0
    upload_errors = 0
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        batch_end = min(i+batch_size, len(points))
        logger.info(f"Upserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}: points {i+1}-{batch_end}")
        
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            successful_uploads += len(batch)
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
            upload_errors += len(batch)
    
    return {
        "successful_uploads": successful_uploads,
        "upload_errors": upload_errors
    }

def print_summary(stats, elapsed_time, logger):
    """Print a summary of the process"""
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("=== Process Summary ===")
    logger.info(f"Total execution time: {minutes} minutes {seconds} seconds")
    logger.info(f"Processed {stats['processed_surahs']}/{stats['total_surahs']} surahs")
    logger.info(f"Successfully processed {stats['processed_verses']}/{stats['total_verses']} verses")
    logger.info(f"Embedding errors: {stats['embedding_errors']}")
    logger.info(f"Successfully uploaded points: {stats['successful_uploads']}")
    logger.info(f"Upload errors: {stats['upload_errors']}")

def main():
    # Setup
    logger = setup_logging()
    logger.info("Starting Quran semantic search embedding process")
    
    # Initialize clients
    openai_client, qdrant_client = initialize_clients(logger)
    
    # Setup collection
    collection_name = "quran_verses"
    setup_collection(qdrant_client, collection_name, logger)
    
    # Load data
    data = load_quran_data("quran_en.json", logger)
    
    # Process data
    start_time = time.time()
    points, embedding_stats = generate_embeddings(data, openai_client, logger)
    
    # Upload to database
    upload_stats = upload_to_qdrant(points, qdrant_client, collection_name, 50, logger)
    
    # Calculate and print summary
    elapsed_time = time.time() - start_time
    stats = {**embedding_stats, **upload_stats}
    print_summary(stats, elapsed_time, logger)
    
    logger.info("Insertion complete!")

if __name__ == "__main__":
    main()
