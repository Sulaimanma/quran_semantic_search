from qdrant_client import QdrantClient
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Connect to Qdrant
try:
    qdrant_client = QdrantClient(host="localhost", port=6333)
    logger.info("Connected to Qdrant")
    
    # Delete the collection
    collection_name = "quran_verses"
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully")
    else:
        logger.info(f"Collection '{collection_name}' does not exist")
        
except Exception as e:
    logger.error(f"Error: {e}")
