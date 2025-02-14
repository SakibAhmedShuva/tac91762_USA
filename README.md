# API Documentation

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently, no authentication is required.

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

### 5. Get Available Voices
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

### 6. Speech-to-Text Conversion
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

### 7. Health Check
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
