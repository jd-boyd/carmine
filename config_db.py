import sqlite3
import json
import os
import time
from datetime import datetime

# Database configuration
DB_PATH = "configs.sqlite"

def init_db():
    """Initialize the SQLite database with the required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for storing configurations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_name TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        config_data TEXT NOT NULL
    )
    ''')
    
    # Create unique index on config_name to prevent duplicates
    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_config_name 
    ON configs (config_name)
    ''')
    
    conn.commit()
    conn.close()

def save_config_to_db(config_name, config_data):
    """
    Save a configuration to the database
    
    Args:
        config_name: Name of the configuration
        config_data: Configuration data as a dictionary
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert config_data to JSON string
        config_json = json.dumps(config_data, indent=4)
        
        # Get current timestamp
        timestamp = int(time.time())
        
        # Check if config with this name already exists
        cursor.execute("SELECT id FROM configs WHERE config_name = ?", (config_name,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing config
            cursor.execute(
                "UPDATE configs SET timestamp = ?, config_data = ? WHERE config_name = ?",
                (timestamp, config_json, config_name)
            )
        else:
            # Insert new config
            cursor.execute(
                "INSERT INTO configs (config_name, timestamp, config_data) VALUES (?, ?, ?)",
                (config_name, timestamp, config_json)
            )
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Error saving configuration to database: {e}")
        return False

def load_config_from_db(config_name):
    """
    Load a configuration from the database
    
    Args:
        config_name: Name of the configuration to load
    
    Returns:
        dict: Configuration data as a dictionary or None if not found or error
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT config_data FROM configs WHERE config_name = ?", 
            (config_name,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Parse JSON string back to dictionary
            return json.loads(result[0])
        else:
            print(f"No configuration found with name: {config_name}")
            return None
    
    except Exception as e:
        print(f"Error loading configuration from database: {e}")
        return None

def delete_config_from_db(config_name):
    """
    Delete a configuration from the database
    
    Args:
        config_name: Name of the configuration to delete
    
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM configs WHERE config_name = ?",
            (config_name,)
        )
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    except Exception as e:
        print(f"Error deleting configuration from database: {e}")
        return False

def list_configs():
    """
    List all configurations in the database
    
    Returns:
        list: List of tuples (config_name, timestamp)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT config_name, timestamp FROM configs ORDER BY config_name"
        )
        
        configs = [(name, timestamp) for name, timestamp in cursor.fetchall()]
        conn.close()
        
        return configs
    
    except Exception as e:
        print(f"Error listing configurations: {e}")
        return []

def export_config_to_file(config_name, file_path):
    """
    Export a configuration from the database to a file
    
    Args:
        config_name: Name of the configuration to export
        file_path: Path to the file to export to
    
    Returns:
        bool: True if exported successfully, False otherwise
    """
    try:
        config = load_config_from_db(config_name)
        if not config:
            return False
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return True
    
    except Exception as e:
        print(f"Error exporting configuration to file: {e}")
        return False

def import_config_from_file(config_name, file_path):
    """
    Import a configuration from a file to the database
    
    Args:
        config_name: Name to save the configuration as
        file_path: Path to the file to import from
    
    Returns:
        bool: True if imported successfully, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        return save_config_to_db(config_name, config)
    
    except Exception as e:
        print(f"Error importing configuration from file: {e}")
        return False

def format_timestamp(timestamp):
    """Format a Unix timestamp as a readable date string"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# Initialize the database when module is imported
init_db()