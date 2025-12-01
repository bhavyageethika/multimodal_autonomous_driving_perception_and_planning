"""
Tag Database

SQLite database for storing and querying auto-tagging results.
Enables searching for specific driving scenarios across videos.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result from a database query."""
    session_id: str
    video_path: str
    frame_idx: int
    timestamp: float
    tags: List[str]
    road_type: str
    maneuver: str
    risk_level: str
    speed_kmh: float
    

class TagDatabase:
    """
    SQLite database for storing and querying auto-tagging results.
    
    Schema:
    - sessions: Metadata about tagging sessions
    - frames: Individual frame tags
    - tags: Tag lookup table
    - frame_tags: Many-to-many relationship
    """
    
    def __init__(self, db_path: str = "tags.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        # check_same_thread=False allows SQLite to work with Streamlit's threading
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_frames INTEGER DEFAULT 0,
                fps REAL DEFAULT 30.0,
                metadata TEXT
            )
        ''')
        
        # Tags lookup table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL,
                tag_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                road_type TEXT,
                road_type_confidence REAL,
                lateral_maneuver TEXT,
                longitudinal_maneuver TEXT,
                turning_maneuver TEXT,
                speed_kmh REAL,
                acceleration REAL,
                risk_level TEXT,
                agent_count INTEGER DEFAULT 0,
                pedestrian_count INTEGER DEFAULT 0,
                vehicle_count INTEGER DEFAULT 0,
                min_ttc REAL,
                closest_distance REAL,
                full_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                UNIQUE(session_id, frame_idx)
            )
        ''')
        
        # Frame-tags junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_tags (
                frame_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                PRIMARY KEY (frame_id, tag_id),
                FOREIGN KEY (frame_id) REFERENCES frames(id),
                FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_road_type ON frames(road_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_risk ON frames(risk_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(tag_name)')
        
        self.conn.commit()
    
    def save_session(self, session_data: Dict) -> str:
        """
        Save a tagging session to the database.
        
        Args:
            session_data: Session metadata dictionary
            
        Returns:
            session_id
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO sessions 
            (session_id, video_path, start_time, end_time, total_frames, fps, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_data.get('session_id'),
            session_data.get('video_path'),
            session_data.get('start_time'),
            session_data.get('end_time'),
            session_data.get('total_frames', 0),
            session_data.get('fps', 30.0),
            json.dumps(session_data)
        ))
        
        self.conn.commit()
        return session_data.get('session_id')
    
    def save_frame_tags(self, session_id: str, frame_tags: Dict) -> int:
        """
        Save tags for a single frame.
        
        Args:
            session_id: Session identifier
            frame_tags: Frame tags dictionary from FrameTags.to_dict()
            
        Returns:
            frame_id in database
        """
        cursor = self.conn.cursor()
        
        # Extract data
        scene = frame_tags.get('scene', {})
        maneuver = frame_tags.get('maneuver', {})
        interaction = frame_tags.get('interaction', {})
        
        # Insert frame
        cursor.execute('''
            INSERT OR REPLACE INTO frames
            (session_id, frame_idx, timestamp, road_type, road_type_confidence,
             lateral_maneuver, longitudinal_maneuver, turning_maneuver,
             speed_kmh, acceleration, risk_level, agent_count,
             pedestrian_count, vehicle_count, min_ttc, closest_distance, full_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            frame_tags.get('frame_idx', 0),
            frame_tags.get('timestamp', 0),
            scene.get('road_type', 'unknown'),
            scene.get('road_type_confidence', 0),
            maneuver.get('lateral', 'lane_keeping'),
            maneuver.get('longitudinal', 'cruising'),
            maneuver.get('turning', 'straight'),
            maneuver.get('speed_kmh', 0),
            maneuver.get('acceleration', 0),
            interaction.get('overall_risk', 'low'),
            interaction.get('agent_count', 0),
            interaction.get('pedestrian_count', 0),
            interaction.get('vehicle_count', 0),
            interaction.get('min_ttc'),
            interaction.get('closest_agent_distance'),
            json.dumps(frame_tags)
        ))
        
        frame_id = cursor.lastrowid
        
        # Insert tags
        all_tags = frame_tags.get('all_tags', [])
        tag_confidences = frame_tags.get('tag_confidences', {})
        
        for tag_name in all_tags:
            # Get or create tag
            cursor.execute(
                'INSERT OR IGNORE INTO tags (tag_name) VALUES (?)',
                (tag_name,)
            )
            cursor.execute(
                'SELECT tag_id FROM tags WHERE tag_name = ?',
                (tag_name,)
            )
            tag_id = cursor.fetchone()[0]
            
            # Link frame to tag
            confidence = tag_confidences.get(tag_name, 1.0)
            cursor.execute('''
                INSERT OR REPLACE INTO frame_tags (frame_id, tag_id, confidence)
                VALUES (?, ?, ?)
            ''', (frame_id, tag_id, confidence))
        
        self.conn.commit()
        return frame_id
    
    def save_all_tags(self, auto_tagger) -> int:
        """
        Save all tags from an AutoTagger instance.
        
        Args:
            auto_tagger: AutoTagger instance with frame_tags
            
        Returns:
            Number of frames saved
        """
        # Save session
        self.save_session(auto_tagger.session.to_dict())
        
        # Save all frames
        count = 0
        for frame_tags in auto_tagger.frame_tags:
            self.save_frame_tags(
                auto_tagger.session.session_id,
                frame_tags.to_dict()
            )
            count += 1
        
        return count
    
    def search_by_tag(self, 
                      tag_name: str,
                      session_id: str = None,
                      limit: int = 100) -> List[QueryResult]:
        """
        Search for frames containing a specific tag.
        
        Args:
            tag_name: Tag to search for
            session_id: Optional session filter
            limit: Maximum results
            
        Returns:
            List of QueryResult objects
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT DISTINCT
                f.session_id, s.video_path, f.frame_idx, f.timestamp,
                f.road_type, f.lateral_maneuver, f.risk_level, f.speed_kmh
            FROM frames f
            JOIN sessions s ON f.session_id = s.session_id
            JOIN frame_tags ft ON f.id = ft.frame_id
            JOIN tags t ON ft.tag_id = t.tag_id
            WHERE t.tag_name = ?
        '''
        params = [tag_name]
        
        if session_id:
            query += ' AND f.session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY f.session_id, f.frame_idx LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append(QueryResult(
                session_id=row['session_id'],
                video_path=row['video_path'],
                frame_idx=row['frame_idx'],
                timestamp=row['timestamp'],
                tags=[tag_name],
                road_type=row['road_type'],
                maneuver=row['lateral_maneuver'],
                risk_level=row['risk_level'],
                speed_kmh=row['speed_kmh']
            ))
        
        return results
    
    def search_by_multiple_tags(self,
                                tags: List[str],
                                match_all: bool = True,
                                session_id: str = None,
                                limit: int = 100) -> List[QueryResult]:
        """
        Search for frames containing multiple tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, all tags must be present
            session_id: Optional session filter
            limit: Maximum results
        """
        cursor = self.conn.cursor()
        
        if match_all:
            # All tags must be present
            placeholders = ','.join(['?' for _ in tags])
            query = f'''
                SELECT 
                    f.session_id, s.video_path, f.frame_idx, f.timestamp,
                    f.road_type, f.lateral_maneuver, f.risk_level, f.speed_kmh
                FROM frames f
                JOIN sessions s ON f.session_id = s.session_id
                WHERE f.id IN (
                    SELECT frame_id 
                    FROM frame_tags ft
                    JOIN tags t ON ft.tag_id = t.tag_id
                    WHERE t.tag_name IN ({placeholders})
                    GROUP BY frame_id
                    HAVING COUNT(DISTINCT t.tag_name) = ?
                )
            '''
            params = tags + [len(tags)]
        else:
            # Any tag matches
            placeholders = ','.join(['?' for _ in tags])
            query = f'''
                SELECT DISTINCT
                    f.session_id, s.video_path, f.frame_idx, f.timestamp,
                    f.road_type, f.lateral_maneuver, f.risk_level, f.speed_kmh
                FROM frames f
                JOIN sessions s ON f.session_id = s.session_id
                JOIN frame_tags ft ON f.id = ft.frame_id
                JOIN tags t ON ft.tag_id = t.tag_id
                WHERE t.tag_name IN ({placeholders})
            '''
            params = tags
        
        if session_id:
            query += ' AND f.session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY f.session_id, f.frame_idx LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append(QueryResult(
                session_id=row['session_id'],
                video_path=row['video_path'],
                frame_idx=row['frame_idx'],
                timestamp=row['timestamp'],
                tags=tags,
                road_type=row['road_type'],
                maneuver=row['lateral_maneuver'],
                risk_level=row['risk_level'],
                speed_kmh=row['speed_kmh']
            ))
        
        return results
    
    def search_high_risk(self, 
                         session_id: str = None,
                         limit: int = 100) -> List[QueryResult]:
        """Search for high-risk frames."""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                f.session_id, s.video_path, f.frame_idx, f.timestamp,
                f.road_type, f.lateral_maneuver, f.risk_level, f.speed_kmh
            FROM frames f
            JOIN sessions s ON f.session_id = s.session_id
            WHERE f.risk_level IN ('high', 'critical')
        '''
        params = []
        
        if session_id:
            query += ' AND f.session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY f.session_id, f.frame_idx LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append(QueryResult(
                session_id=row['session_id'],
                video_path=row['video_path'],
                frame_idx=row['frame_idx'],
                timestamp=row['timestamp'],
                tags=['high_risk'],
                road_type=row['road_type'],
                maneuver=row['lateral_maneuver'],
                risk_level=row['risk_level'],
                speed_kmh=row['speed_kmh']
            ))
        
        return results
    
    def get_tag_statistics(self, session_id: str = None) -> Dict:
        """Get statistics about tags in the database."""
        cursor = self.conn.cursor()
        
        # Get tag counts
        if session_id:
            cursor.execute('''
                SELECT t.tag_name, COUNT(*) as count
                FROM tags t
                JOIN frame_tags ft ON t.tag_id = ft.tag_id
                JOIN frames f ON ft.frame_id = f.id
                WHERE f.session_id = ?
                GROUP BY t.tag_name
                ORDER BY count DESC
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT t.tag_name, COUNT(*) as count
                FROM tags t
                JOIN frame_tags ft ON t.tag_id = ft.tag_id
                GROUP BY t.tag_name
                ORDER BY count DESC
            ''')
        
        tag_counts = {row['tag_name']: row['count'] for row in cursor.fetchall()}
        
        # Get session count
        cursor.execute('SELECT COUNT(*) FROM sessions')
        session_count = cursor.fetchone()[0]
        
        # Get frame count
        if session_id:
            cursor.execute(
                'SELECT COUNT(*) FROM frames WHERE session_id = ?',
                (session_id,)
            )
        else:
            cursor.execute('SELECT COUNT(*) FROM frames')
        frame_count = cursor.fetchone()[0]
        
        # Get risk distribution
        if session_id:
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM frames WHERE session_id = ?
                GROUP BY risk_level
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM frames GROUP BY risk_level
            ''')
        risk_dist = {row['risk_level']: row['count'] for row in cursor.fetchall()}
        
        return {
            'session_count': session_count,
            'frame_count': frame_count,
            'tag_counts': tag_counts,
            'risk_distribution': risk_dist,
            'unique_tags': len(tag_counts)
        }
    
    def get_sessions(self) -> List[Dict]:
        """Get all tagging sessions."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT session_id, video_path, start_time, total_frames, fps
            FROM sessions ORDER BY start_time DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def export_session(self, session_id: str, format: str = 'json') -> Any:
        """
        Export all data for a session.
        
        Args:
            session_id: Session to export
            format: 'json' or 'csv'
        """
        cursor = self.conn.cursor()
        
        # Get session
        cursor.execute(
            'SELECT * FROM sessions WHERE session_id = ?',
            (session_id,)
        )
        session = dict(cursor.fetchone())
        
        # Get frames with full data
        cursor.execute(
            'SELECT full_data FROM frames WHERE session_id = ? ORDER BY frame_idx',
            (session_id,)
        )
        frames = [json.loads(row['full_data']) for row in cursor.fetchall()]
        
        if format == 'json':
            return json.dumps({
                'session': session,
                'frames': frames
            }, indent=2)
        elif format == 'csv':
            # Return list of dicts for CSV export
            return frames
        
        return None
    
    def delete_session(self, session_id: str):
        """Delete a tagging session and all its data."""
        cursor = self.conn.cursor()
        
        # Delete frame tags
        cursor.execute('''
            DELETE FROM frame_tags WHERE frame_id IN (
                SELECT id FROM frames WHERE session_id = ?
            )
        ''', (session_id,))
        
        # Delete frames
        cursor.execute('DELETE FROM frames WHERE session_id = ?', (session_id,))
        
        # Delete session
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        self.close()

