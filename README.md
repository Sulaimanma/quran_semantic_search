# Quran Semantic Search Engine

A vector-based semantic search engine for the Quran that enables meaningful search through verses using natural language understanding.

## Overview

This project creates a semantic search capability for Quran verses by:
1. Loading the Quran text and translations from JSON
2. Converting each verse into vector embeddings using OpenAI's embedding model
3. Storing these embeddings in Qdrant vector database
4. Enabling semantic similarity search across verses

## Prerequisites

- Python 3.8+
- Qdrant server running (default: localhost:6333)
- OpenAI API key
- Required Python packages (see Installation)

## Installation

1. Clone this repository
```bash
git clone https://github.com/Sulaimanma/quran-semantic-search.git
cd quran-semantic-search
```

2. Install required packages
```bash
pip install openai qdrant-client python-dotenv tqdm
```

3. Set up environment variables
Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. Make sure Qdrant is running (using Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Generating Embeddings and Populating the Database

Run the main script to process the Quran data and populate the vector database:

```bash
python main.py
```

This will:
- Load the Quran data from JSON
- Generate embeddings for each verse
- Store the embeddings in Qdrant

### Clearing the Database

If you need to reset and delete the collection:

```bash
python delete_collection.py
```

## How It Works

1. **Data Loading**: The system loads Quran data from `quran_en.json` containing all chapters (surahs) and verses with English translations.

2. **Embedding Generation**: For each verse, an embedding is generated using OpenAI's embedding model (default: text-embedding-ada-002).

3. **Vector Storage**: Embeddings are stored in Qdrant vector database along with metadata (surah number, verse number, Arabic text, English translation).

4. **Semantic Search**: (Implementation for searching can be added) Users can search using natural language, and the system will find semantically similar verses.

## Project Structure

- `main.py`: Main script for processing Quran data and generating embeddings
- `delete_collection.py`: Utility to delete the Qdrant collection
- `quran_en.json`: Source data containing Quran text and translations

## Credits

- Quran JSON data: [risan/quran-json](https://github.com/risan/quran-json/blob/main/dist/quran_en.json)
- Vector Database: [Qdrant](https://qdrant.tech/)
- Embeddings: [OpenAI](https://openai.com/)

## License

MIT License
