# mariadb:11.7-rc
# Combines semantic (70%) and geographic (30%) scores

import os
import json
import io
import pymysql
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weasyprint import HTML
from flask_cors import CORS
import uuid
from datetime import datetime
import logging
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask import Response, stream_with_context
from werkzeug.utils import secure_filename
import jinja2
from tts_engine import *
from stt_engine import *

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logger = logging.getLogger(__name__)

load_dotenv()

CONFIG = {
    "database": {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "name": os.getenv("DB_NAME"),
        "table": os.getenv("TABLE_NAME", "embeddings")
    },
    "model": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": int(os.getenv("BATCH_SIZE", 32))
    },
    "paths": {
        "data": r".\dataset\10172 CHRISTOPHER ST_ CYPRESS.csv",
        "index": "property_index.faiss",
        "report": "search_results_cl-custom.pdf",
        "results_folder": "./results"
    },
    "search": {
        "query": os.getenv("SEARCH_QUERY", "Modern house with pool"),
        "k": int(os.getenv("SEARCH_K", 5))
    }
}

CONFIG["paths"]["results_folder"] = "./results"

os.makedirs(CONFIG["paths"]["results_folder"], exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('property_search.log'),
        logging.StreamHandler()
    ]
)

class SearchEngine:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config["model"]["name"])
        self.index = None
        self.property_ids = None
        self.table_name = config["database"]["table"]

    def process_data(self, df, text_columns):
        """Process dataframe and create embeddings for the specified text columns"""
        df['combined_text'] = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            df['combined_text'].tolist(),
            batch_size=self.config["model"]["batch_size"],
            show_progress_bar=True
        )
        return embeddings
        
    def validate_search_params(self, query, lat, lon, k, radius_km):
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(radius_km, (int, float)) or radius_km <= 0:
            raise ValueError("radius_km must be a positive number")
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")

    def build_index(self, embeddings, property_ids):
        """Build FAISS index from embeddings with corresponding property IDs"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.property_ids = property_ids  # Store matching property IDs
        faiss.write_index(self.index, self.config["paths"]["index"])
        logger.info(f"Built FAISS index with {embeddings.shape[0]} embeddings")

    def load_index(self):
        if os.path.exists(self.config["paths"]["index"]):
            logger.info(f"Loading existing index from {self.config['paths']['index']}")
            self.index = faiss.read_index(self.config["paths"]["index"])
        return self.index

    def search(self, query, k=5):
        """Search for similar properties using FAISS"""
        if self.index is None or self.property_ids is None:
            raise ValueError("Index not properly initialized. Call build_index first.")

        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        # Map FAISS indices to actual property IDs
        result_ids = [self.property_ids[i] for i in indices[0]]
        return distances[0], result_ids

    def hybrid_search(self, query, lat, lon, k=5, radius_km=10):
        self.validate_search_params(query, lat, lon, k, radius_km)
        query_embed = self.model.encode(query)

        with DatabaseManager(self.config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                SELECT *,
                    ST_Distance_Sphere(
                        location,
                        ST_PointFromText(%s)
                    ) / 1000 as distance_km,
                    COALESCE(VECTOR_DOT_PRODUCT(embedding, %s), 0) as semantic_score
                FROM {self.table_name}
                WHERE ST_Distance_Sphere(
                    location,
                    ST_PointFromText(%s)
                ) <= %s
                ORDER BY (semantic_score * 0.7) + (1 / (1 + distance_km / {radius_km}) * 0.3) DESC
                LIMIT %s
                """, (
                    f'POINT({lon} {lat})',
                    json.dumps(query_embed.tolist()),
                    f'POINT({lon} {lat})',
                    radius_km * 1000,  # Convert to meters
                    k
                ))
                results = cursor.fetchall()

                # Calculate combined score for each result
                for result in results:
                    semantic_score = result.pop('semantic_score')
                    distance_km = result.pop('distance_km')
                    result['combined_score'] = (semantic_score * 0.7) + (1 / (1 + distance_km / radius_km) * 0.3)

                return results

# Global components
search_engine = None
if os.path.exists(CONFIG["paths"]["index"]):
    search_engine = SearchEngine(CONFIG)
    search_engine.load_index()
    logger.info("Loaded existing FAISS index")

class DatabaseManager:
    def __init__(self, config):
        self.config = config["database"]
        self.connection = None
        self.table_name = self.config["table"]

    def validate_coordinates(self, latitude, longitude):
        """Validate latitude and longitude values"""
        if latitude is None or longitude is None:
            raise ValueError("Latitude and longitude cannot be NULL")
        try:
            lat, lon = float(latitude), float(longitude)
            if not (-90 <= lat <= 90):
                raise ValueError(f"Invalid latitude value: {latitude}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Invalid longitude value: {longitude}")
        except ValueError as e:
            raise ValueError(f"Invalid coordinate format: {str(e)}")

    def validate_connection(self):
        """Validate database connection parameters"""
        required_params = ['host', 'port', 'user', 'password', 'name']
        missing_params = [param for param in required_params 
                         if not self.config.get(param)]
        if missing_params:
            raise ValueError(f"Missing database configuration parameters: "
                           f"{', '.join(missing_params)}")

    def create_vector_dot_product_function(self):
        """Create a function for vector dot product calculation"""
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("DROP FUNCTION IF EXISTS VECTOR_DOT_PRODUCT")
                cursor.execute("""
                CREATE FUNCTION VECTOR_DOT_PRODUCT(vec1 JSON, vec2 JSON)
                RETURNS FLOAT
                DETERMINISTIC
                BEGIN
                    DECLARE i INT DEFAULT 0;
                    DECLARE n INT;
                    DECLARE dot_product FLOAT DEFAULT 0.0;
                    
                    IF JSON_LENGTH(vec1) != JSON_LENGTH(vec2) THEN
                        SIGNAL SQLSTATE '45000'
                        SET MESSAGE_TEXT = 'Vector dimensions do not match';
                    END IF;
                    
                    SET n = JSON_LENGTH(vec1);
                    
                    WHILE i < n DO
                        SET dot_product = dot_product + 
                            (JSON_EXTRACT(vec1, CONCAT('$[', i, ']')) * 
                            JSON_EXTRACT(vec2, CONCAT('$[', i, ']')));
                        SET i = i + 1;
                    END WHILE;
                    
                    RETURN dot_product;
                END
                """)
            except pymysql.Error as e:
                logger.error(f"Error creating vector dot product function: {str(e)}")
                raise

    def create_tables(self):
        with self.connection.cursor() as cursor:
            try:
                self.create_vector_dot_product_function()
                logger.info("VECTOR_DOT_PRODUCT function created successfully")
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                cursor.execute(f"DROP TRIGGER IF EXISTS update_location_{self.table_name}")
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    property_id INT PRIMARY KEY AUTO_INCREMENT,
                    description TEXT NOT NULL,
                    embedding JSON NOT NULL,
                    latitude DECIMAL(10, 8) NOT NULL,
                    longitude DECIMAL(11, 8) NOT NULL,
                    location POINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    SPATIAL INDEX(location),
                    FULLTEXT INDEX(description)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                logger.info(f"Table {self.table_name} created successfully")
                self.connection.commit()
            except pymysql.Error as e:
                self.connection.rollback()
                logger.error(f"Error creating tables and triggers: {str(e)}")
                raise
                
                cursor.execute(f"""
                CREATE TRIGGER update_location_{self.table_name} BEFORE INSERT ON {self.table_name}
                FOR EACH ROW
                BEGIN
                    IF NEW.latitude IS NULL OR NEW.longitude IS NULL THEN
                        SIGNAL SQLSTATE '45000'
                        SET MESSAGE_TEXT = 'Latitude and longitude cannot be NULL';
                    END IF;
                    SET NEW.location = ST_PointFromText(
                        CONCAT('POINT(', NEW.longitude, ' ', NEW.latitude, ')')
                    );
                END
                """)
                self.connection.commit()
                logger.info("Successfully created tables and triggers")
            except pymysql.Error as e:
                self.connection.rollback()
                logger.error(f"Error creating tables and triggers: {str(e)}")
                raise

    def ensure_table_exists(self):
        """Ensure that the table exists, create it if it doesn't"""
        with self.connection.cursor() as cursor:
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                property_id INT PRIMARY KEY AUTO_INCREMENT,
                description TEXT NOT NULL,
                embedding JSON NOT NULL,
                latitude DECIMAL(10, 8) NOT NULL,
                longitude DECIMAL(11, 8) NOT NULL,
                location POINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                SPATIAL INDEX(location),
                FULLTEXT INDEX(description)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            self.connection.commit()
            logger.info(f"Ensured table {self.table_name} exists")

    def __enter__(self):
        self.validate_connection()
        try:
            logger.info("Connecting to database %s...", self.config['name'])
            self.connection = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["name"],
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info("Database connection successful")
            # Add these lines:
            self.ensure_table_exists()
            self.create_vector_dot_product_function()
            return self.connection
        except Exception as e:
            logger.error("Database connection error: %s", str(e))
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config["model"]["name"])
        self.primary_columns = [
            'PrivateRemarks', 'PublicRemarks', 'BuildingFeatures',
            'InteriorFeatures', 'ExteriorFeatures', 'ArchitecturalStyle',
            'SpecialListingConditions', 'ZoningDescription'
        ]
        self.secondary_columns = [
            'PoolFeatures', 'CommunityFeatures', 'FireplaceFeatures'
        ]

    def load_data(self):
        file_path = self.config["paths"]["data"]
        logger.info(f"Attempting to load data from: {file_path}")
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist at path: {file_path}")
                raise FileNotFoundError(f"Dataset not found: {file_path}")
            
            df = pd.read_csv(file_path)
            required = ['Latitude', 'Longitude'] + self.primary_columns
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            if df['Latitude'].isnull().any() or df['Longitude'].isnull().any():
                raise ValueError("Latitude/Longitude contains null values")
            if (df['Latitude'].abs() > 90).any():
                raise ValueError("Latitude values must be between -90 and 90")
            if (df['Longitude'].abs() > 180).any():
                raise ValueError("Longitude values must be between -180 and 180")
                
            return df.fillna('')
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def combine_text_features(self, row):
        text = " ".join(str(row[col]) for col in self.primary_columns if col in row)
        secondary_text = " ".join(str(row[col]) for col in self.secondary_columns if col in row)
        return f"{text} {secondary_text}"

    def generate_embeddings(self, df):
        logger.info(f"Generating embeddings using model: {self.config['model']['name']}")
        df = df.copy()
        df['combined_text'] = df.apply(self.combine_text_features, axis=1)
        texts = df['combined_text'].tolist()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config["model"]["batch_size"],
            show_progress_bar=True
        )
        
        df['embedding'] = embeddings.tolist()
        return df

class ReportGenerator:
    @staticmethod
    def generate_pdf_bytes(query, results, lat=None, lon=None):
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
                
                body {
                    font-family: 'Roboto', sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }
                
                .header {
                    margin-bottom: 40px;
                    border-bottom: 2px solid #2196F3;
                    padding-bottom: 20px;
                }
                
                .header h1 {
                    color: #1976D2;
                    font-size: 28px;
                    margin-bottom: 10px;
                }
                
                .search-info {
                    background-color: #E3F2FD;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                
                .search-info p {
                    margin: 5px 0;
                }
                
                .timestamp {
                    color: #666;
                    font-size: 0.9em;
                    font-style: italic;
                }
                
                .results-container {
                    display: grid;
                    gap: 25px;
                    margin-top: 0;
                }
                
                .result {
                    margin-top: 0;
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .result h3 {
                    color: #1976D2;
                    margin-top: 0;
                    font-size: 20px;
                    border-bottom: 1px solid #E0E0E0;
                    padding-bottom: 10px;
                }
                
                .property-details {
                    margin: 15px 0;
                }
                
                .property-details strong {
                    color: #455A64;
                }
                
                .score-badge {
                    display: inline-block;
                    background-color: #4CAF50;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.9em;
                }
                
                .location-info {
                    background-color: #FAFAFA;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-size: 0.9em;
                }
                
                .meta-info {
                    color: #757575;
                    font-size: 0.8em;
                    margin-top: 10px;
                    border-top: 1px dashed #E0E0E0;
                    padding-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Property Search Results</h1>
                <div class="search-info">
                    <p><strong>Search Query:</strong> "{{ query }}"</p>
                    {% if lat and lon %}
                    <p><strong>Search Location:</strong> 
                        Latitude: {{ "%.6f"|format(lat) }}°, 
                        Longitude: {{ "%.6f"|format(lon) }}°
                    </p>
                    {% endif %}
                    <p class="timestamp">Generated on: {{ timestamp }}</p>
                </div>
            </div>
            
            <div class="results-container">
                {% for result in results %}
                <div class="result">
                    <h3>Property #{{ loop.index }}</h3>
                    <div class="property-details">
                        <p><strong>Description:</strong><br>{{ result.description }}</p>
                    </div>
                    <div class="score-badge">
                        Similarity Score: {{ result.score }}
                    </div>
                    <div class="location-info">
                        <p><strong>Location:</strong> {{ result.latitude }}°N, {{ result.longitude }}°E</p>
                    </div>
                    <div class="meta-info">
                        <p>Created: {{ result.created_at }} | Last Updated: {{ result.updated_at }}</p>
                    </div>
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        
        results_data = []
        for result in results:
            results_data.append({
                'description': result['description'],
                'score': f"{result['similarity_score']:.3f}",
                'latitude': f"{result['latitude']:.6f}",
                'longitude': f"{result['longitude']:.6f}",
                'created_at': result['created_at'],
                'updated_at': result['updated_at']
            })
        
        html_content = jinja2.Template(template).render(
            query=query,
            lat=lat,
            lon=lon,
            results=results_data,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return HTML(string=html_content).write_pdf()

# Global components
search_engine = None
if os.path.exists(CONFIG["paths"]["index"]):
    search_engine = SearchEngine(CONFIG)
    search_engine.load_index()
    logger.info("Loaded existing FAISS index")

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    global search_engine
    try:
        # Get the clean_start parameter from the request, default to False
        clean_start = request.args.get('clean_start', 'false').lower() == 'true'
        
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the uploaded file temporarily
        file_path = os.path.join(CONFIG["paths"]["results_folder"], file.filename)
        file.save(file_path)
        
        # Update the data path in the config
        CONFIG["paths"]["data"] = file_path
        
        processor = DataProcessor(CONFIG)
        df = processor.load_data()
        df = processor.generate_embeddings(df)
        
        db_manager = DatabaseManager(CONFIG)
        successful_property_ids = []
        successful_embeddings = []

        with db_manager as conn:
            if clean_start:
                db_manager.create_tables()
            with conn.cursor() as cursor:
                for _, row in df.iterrows():
                    try:
                        db_manager.validate_coordinates(row['Latitude'], row['Longitude'])
                        cursor.execute(f"""
                        INSERT INTO {db_manager.table_name}
                        (description, embedding, latitude, longitude, location)
                        VALUES (%s, %s, %s, %s, ST_PointFromText(CONCAT('POINT(', %s, ' ', %s, ')')))
                        """, (
                            row['combined_text'],
                            json.dumps(row['embedding']),
                            row['Latitude'],
                            row['Longitude'],
                            row['Longitude'],
                            row['Latitude']
                        ))
                        property_id = cursor.lastrowid
                        successful_property_ids.append(property_id)
                        successful_embeddings.append(row['embedding'])
                    except ValueError as e:
                        logger.warning(f"Skipping row due to invalid coordinates: {e}")
                        continue
                conn.commit()

        # Build index with only successful entries
        embeddings_array = np.array(successful_embeddings)
        search_engine = SearchEngine(CONFIG)
        search_engine.build_index(embeddings_array, successful_property_ids)
        
        return jsonify({
            "message": "System initialized successfully",
            "properties_processed": len(successful_property_ids),
            "index_size": embeddings_array.shape[0]
        }), 200
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_properties():
    global search_engine
    try:
        if not search_engine or not search_engine.property_ids:
            raise ValueError("Search engine not properly initialized")

        data = request.get_json()
        query = data.get('query', CONFIG["search"]["query"])
        lat = data.get('latitude')
        lon = data.get('longitude')
        k = int(data.get('k', CONFIG["search"]["k"]))
        radius_km = data.get('radius_km', 10)  # Default radius
        generate_pdf = data.get('generate_pdf', False)

        # Validate and perform appropriate search
        if lat and lon:
            # Use hybrid search with geographic filtering
            try:
                search_engine.validate_search_params(query, float(lat), float(lon), k, float(radius_km))
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            results = search_engine.hybrid_search(
                query=query,
                lat=float(lat),
                lon=float(lon),
                k=k,
                radius_km=float(radius_km)
            )
            formatted_results = [
                {
                    'property_id': res['property_id'],
                    'description': res['description'],
                    'latitude': float(res['latitude']),
                    'longitude': float(res['longitude']),
                    'similarity_score': res['combined_score'],
                    'created_at': res['created_at'].isoformat(),
                    'updated_at': res['updated_at'].isoformat()
                } for res in results
            ]
        else:
            # Use FAISS-based semantic search
            distances, property_ids = search_engine.search(query, k)
            
            with DatabaseManager(CONFIG) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    SELECT * FROM {CONFIG['database']['table']}
                    WHERE property_id IN ({','.join(['%s']*len(property_ids))})
                    """, tuple(property_ids))
                    results = cursor.fetchall()

            id_to_index = {pid: idx for idx, pid in enumerate(property_ids)}
            results.sort(key=lambda x: id_to_index[x['property_id']])
            
            formatted_results = [
                {
                    'property_id': res['property_id'],
                    'description': res['description'],
                    'latitude': float(res['latitude']),
                    'longitude': float(res['longitude']),
                    'similarity_score': 1 / (1 + distances[idx]),
                    'created_at': res['created_at'].isoformat(),
                    'updated_at': res['updated_at'].isoformat()
                } for idx, res in enumerate(results)
            ]

        if generate_pdf:
            current_date = datetime.now().strftime("%Y%m%d")
            unique_id = uuid.uuid4().hex[:8]
            pdf_filename = f"search_result_{current_date}_{unique_id}.pdf"
            pdf_path = os.path.join(CONFIG["paths"]["results_folder"], pdf_filename)

            pdf_bytes = ReportGenerator.generate_pdf_bytes(
                query=query,
                results=formatted_results,
                lat=float(lat) if lat else None,
                lon=float(lon) if lon else None
            )
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)

            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=pdf_filename
            )
            
        return jsonify({"results": formatted_results}), 200

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 400

#tts
@app.route('/api/tts/convert', methods=['POST'])
def convert_text():
    """
    Convert text to speech using the TTS service.
    
    Expected JSON payload:
    {
        "text": "Text to convert to speech",
        "voice": "female_1",  # optional, defaults to female_1
        "language": "en-US"   # optional, defaults to en-US
    }
    """
    try:
        # Validate request data
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Validate text input
        text = data.get('text')
        if not text or not isinstance(text, str):
            return jsonify({'error': 'Valid text string is required'}), 400
            
        # Validate voice selection
        voice = data.get('voice', 'female_1')
        available_voices = tts_service.get_available_voices()
        voice_ids = [v['id'] for v in available_voices['voices']]
        
        if voice not in voice_ids:
            return jsonify({
                'error': f'Invalid voice. Available voices: {voice_ids}'
            }), 400
            
        # Get language (optional parameter)
        language = data.get('language', 'en-US')
        
        # Convert text to speech
        task_id = tts_service.convert_text_to_speech(text, voice, language)
        if not task_id:
            return jsonify({'error': 'TTS conversion failed'}), 500
            
        # Return success response with task information
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'status_url': f'/api/tts/status/{task_id}'
        }), 202
            
    except Exception as e:
        logger.error(f"TTS convert error: {str(e)}")
        return jsonify({'error': str(e)}), 500
            
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'status_url': f'/api/tts/status/{task_id}'
        }), 202
            
    except Exception as e:
        logger.error(f"TTS convert error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
        language = data.get('language', 'en-US')
        
        task_id = tts_service.convert_text_to_speech(text, voice, language)
        if not task_id:
            return jsonify({'error': 'TTS conversion failed'}), 500
            
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'status_url': f'/api/tts/status/{task_id}'
        }), 202
            
    except Exception as e:
        logger.error(f"TTS convert error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/status/<task_id>', methods=['GET'])
def check_status(task_id):
    try:
        status = tts_service.check_status(task_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/save', methods=['POST'])
def save_speech():
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'female_1')
        language = data.get('language', 'en-US')
        format = data.get('format', 'wav')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
            
        result = tts_service.save_speech_file(text, voice, language, format)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Save failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/voices', methods=['GET'])
def get_voices():
    try:
        voices = tts_service.get_available_voices()
        return jsonify(voices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/stream', methods=['POST'])
def stream_speech():
    try:
        data = request.json
        text = data.get('text')
        voice = data.get('voice', 'female_1')
        language = data.get('language', 'en-US')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        def generate():
            for chunk in tts_service.stream_speech(text, voice, language):
                if chunk is not None:
                    # Convert numpy array to bytes
                    chunk_bytes = chunk.tobytes()
                    yield chunk_bytes

        return Response(stream_with_context(generate()), 
                       mimetype='audio/wav')
                       
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/download/<task_id>', methods=['GET'])
def download_audio(task_id):
    try:
        status = tts_service.check_status(task_id)
        
        if status['status'] != 'completed':
            return jsonify({'error': 'Audio not ready or doesn\'t exist'}), 404
            
        return send_file(status['audio_path'], 
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'{task_id}.wav')
                        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#stt
@app.route('/api/stt/convert', methods=['POST'])
def transcribe_audio():
    try:
        # Check if audio file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Transcribe the audio
        transcription = transcriber.transcribe(filepath)

        # Clean up - remove the uploaded file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'transcription': transcription
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)