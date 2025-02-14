# API Documentation

Python 3.10.11 is preferred. Please use "postman_collection.json".

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently, no authentication is required.
 
# Environment Variables Documentation (.env)

## Database Configuration

### Required Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DB_HOST` | Database server hostname | - | `localhost` |
| `DB_PORT` | Database server port | `3306` | `3306` |
| `DB_USER` | Database username | - | `root` |
| `DB_PASSWORD` | Database password | - | `12345` |
| `DB_NAME` | Database name | - | `properties_db` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `TABLE_NAME` | Name of the embeddings table | `embeddings` | `embeddings` |
| `BATCH_SIZE` | Batch size for processing embeddings | `32` | `32` |

## Commented Variables (Currently Inactive)

These variables are commented out in the current configuration but can be uncommented and configured as needed:

```plaintext
# CSV_PATH - Path to the dataset CSV file
# SEARCH_QUERY - Default search query for testing
# SEARCH_K - Default number of results to return
```

## Endpoints

### 1. Initialize System
```http
POST /initialize
```
Initializes the search system with property data.

**Parameters:**
- `clean_start` (query, boolean, optional): If true, recreates tables
- `file` (form-data, required): CSV file containing property data

**Response:**
```json
{
    "message": "System initialized successfully",
    "properties_processed": 100,
    "index_size": 100
}
```

### 2. Search Properties
```http
POST /search
```
Search for properties using semantic and/or geographic criteria.

**Request Body:**
```json
{
    "query": "Modern house with pool",
    "latitude": 34.0522,
    "longitude": -118.2437,
    "k": 5,
    "radius_km": 10,
    "generate_pdf": false
}
```

**Response:**
```json
{
    "results": [
        {
            "property_id": 1,
            "description": "Modern house...",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "similarity_score": 0.85,
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00"
        }
    ]
}
```

### 3. Text-to-Speech Conversion
```http
POST /tts/convert
```
Convert text to speech.

**Request Body:**
```json
{
    "text": "Text to convert",
    "voice": "female_1",
    "language": "en-US"
}
```

**Response:**
```json
{
    "task_id": "abc123",
    "status": "processing",
    "status_url": "/api/tts/status/abc123"
}
```

### 4. Check TTS Status
```http
GET /tts/status/<task_id>
```
Check the status of a TTS conversion task.

**Response:**
```json
{
    "status": "completed",
    "audio_path": "/path/to/audio.wav"
}
```

### 5. Save Speech File
```http
POST /tts/save
```
Convert text to speech and save as an audio file.

**Request Body:**
```json
{
    "text": "Text to convert to speech",
    "voice": "female_1",
    "language": "en-US",
    "format": "wav"
}
```

**Response:**
```json
{
    "success": true,
    "file_path": "/path/to/saved/audio.wav",
    "duration": 2.5,
    "format": "wav"
}
```
### 6. Get Available Voices
```http
GET /tts/voices
```
Get list of available TTS voices.

**Response:**
```json
{
    "voices": [
        {
            "id": "female_1",
            "name": "Female Voice 1",
            "language": "en-US"
        }
    ]
}
```

### 7. Stream Speech Audio
```http
POST /tts/stream
```
Stream audio directly to the client.

**Request Body:**
```json
{
    "text": "Text to convert to speech",
    "voice": "female_1",
    "language": "en-US"
}
```

**Response:**
- Content-Type: audio/wav
- Streams audio data directly to client

### 8. Download Audio File
```http
GET /tts/download/<task_id>
```
Download the generated audio file for a completed TTS task.

**Parameters:**
- `task_id` (path, required): The ID of the completed TTS task

**Response:**
- Content-Type: audio/wav
- Downloads the audio file

These endpoints complete your TTS functionality with options for:
- Saving audio files
- Streaming audio directly
- Downloading generated audio files

The implementation includes error handling and supports various audio formats. The streaming endpoint is particularly useful for real-time applications where immediate audio playback is needed.

Example usage with curl:

```bash
# Save audio file
curl -X POST http://localhost:5000/api/tts/save \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice":"female_1","format":"wav"}'

# Stream audio
curl -X POST http://localhost:5000/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice":"female_1"}' \
  --output output.wav

# Download audio file
curl -X GET http://localhost:5000/api/tts/download/task123 \
  --output downloaded_audio.wav
```

Error Handling:
- 400: Invalid parameters or missing data
- 404: Audio file not found
- 500: Server processing error

The streaming endpoint uses chunked transfer encoding to deliver audio data as it's generated, making it suitable for real-time applications.

### 9. Speech-to-Text Conversion
```http
POST /stt/convert
```
Convert audio to text.

**Parameters:**
- `file` (form-data, required): Audio file (WAV, MP3, or OGG)

**Response:**
```json
{
    "success": true,
    "transcription": "Transcribed text..."
}
```

### 10. Health Check
```http
GET /health
```
Check API health status.

**Response:**
```json
{
    "status": "healthy"
}
```

## Error Responses
All endpoints may return the following error response:

```json
{
    "error": "Error message description"
}
```

## Status Codes
- 200: Success
- 202: Accepted (for async operations)
- 400: Bad Request
- 500: Internal Server Error

## File Format Support
- Audio Input: .wav, .mp3, .ogg
- Data Input: .csv
This documentation provides a comprehensive overview of your API's endpoints and functionality. You may want to add more specific details about:
- Rate limiting
- Authentication (when implemented)
- Specific error codes and messages
- Data schemas
- Performance expectations
- Usage examples
