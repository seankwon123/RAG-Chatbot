# database_manager.py
"""
PostgreSQL database manager for the Bitovi RAG system.
Handles all database operations including fetching articles and keyword searches.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from contextlib import contextmanager

from models import Article, RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and queries"""
    
    def __init__(self, config: RAGConfig = None):
        """Initialize with configuration"""
        self.config = config or RAGConfig
        self.connection_params = self.config.get_postgres_config()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test the database connection and show article count"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Test basic connection
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    
                    # Get article count
                    cursor.execute("SELECT COUNT(*) FROM articles")
                    count = cursor.fetchone()[0]
                    
                    # Get date range
                    cursor.execute("""
                        SELECT 
                            MIN(published_date) as earliest,
                            MAX(published_date) as latest
                        FROM articles
                        WHERE published_date IS NOT NULL
                    """)
                    dates = cursor.fetchone()
                    
                    logger.info(f"✅ Database connected successfully")
                    logger.info(f"📊 Found {count} articles in database")
                    if dates[0] and dates[1]:
                        logger.info(f"📅 Date range: {dates[0].date()} to {dates[1].date()}")
                    
                    return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_all_articles(self, limit: Optional[int] = None) -> List[Article]:
        """
        Retrieve all articles from the database.
        Orders by published_date DESC to get newest first.
        """
        query = """
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash
            FROM articles
            ORDER BY published_date DESC NULLS LAST
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        articles = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        articles.append(self._row_to_article(row))
            
            logger.info(f"Retrieved {len(articles)} articles from database")
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            raise
        
        return articles
    
    def get_article_by_id(self, article_id: int) -> Optional[Article]:
        """Get a single article by ID"""
        query = """
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash
            FROM articles
            WHERE id = %s
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, (article_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_article(row)
        except Exception as e:
            logger.error(f"Error getting article {article_id}: {e}")
        
        return None
    
    def get_articles_by_ids(self, article_ids: List[int]) -> List[Article]:
        """Get multiple articles by their IDs"""
        if not article_ids:
            return []
        
        placeholders = ",".join(["%s"] * len(article_ids))
        query = f"""
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash
            FROM articles
            WHERE id IN ({placeholders})
            ORDER BY published_date DESC NULLS LAST
        """
        
        articles = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, article_ids)
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        articles.append(self._row_to_article(row))
        except Exception as e:
            logger.error(f"Error getting articles by IDs: {e}")
        
        return articles
    
    def get_latest_articles(self, limit: int = 10) -> List[Article]:
        """Get the most recent articles by published date"""
        query = """
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash
            FROM articles
            WHERE published_date IS NOT NULL
            ORDER BY published_date DESC
            LIMIT %s
        """
        
        articles = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, (limit,))
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        articles.append(self._row_to_article(row))
            
            logger.info(f"Retrieved {len(articles)} latest articles")
            
        except Exception as e:
            logger.error(f"Error getting latest articles: {e}")
        
        return articles
    
    def search_articles_by_keyword(self, keyword: str) -> List[Article]:
        """
        Full-text search across title, excerpt, content, tags, and author.
        Uses PostgreSQL's ILIKE for case-insensitive search.
        Returns results ordered by relevance (title matches first, then tags, etc.)
        """
        query = """
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash,
                -- Calculate a simple relevance score
                CASE 
                    WHEN title ILIKE %s THEN 5
                    WHEN tags::text ILIKE %s THEN 4
                    WHEN excerpt ILIKE %s THEN 3
                    WHEN author ILIKE %s THEN 2
                    ELSE 1
                END as relevance
            FROM articles
            WHERE 
                title ILIKE %s OR
                excerpt ILIKE %s OR
                content ILIKE %s OR
                tags::text ILIKE %s OR
                author ILIKE %s
            ORDER BY 
                relevance DESC,
                published_date DESC NULLS LAST
            LIMIT 50
        """
        
        search_pattern = f"%{keyword}%"
        # 4 for CASE statement + 5 for WHERE clause = 9 parameters
        params = [search_pattern] * 9
        
        articles = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        articles.append(self._row_to_article(row))
            
            logger.info(f"Found {len(articles)} articles matching '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error searching for keyword '{keyword}': {e}")
        
        return articles
    
    def get_articles_by_date_range(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> List[Article]:
        """Get articles within a date range"""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("published_date >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("published_date <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT 
                id, title, url, content, excerpt, author,
                published_date, tags, word_count,
                created_at, updated_at, content_hash
            FROM articles
            WHERE {where_clause}
            ORDER BY published_date DESC
        """
        
        articles = []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        articles.append(self._row_to_article(row))
                        
        except Exception as e:
            logger.error(f"Error getting articles by date range: {e}")
        
        return articles
    
    def get_article_stats(self) -> Dict[str, Any]:
        """Get statistics about the articles in the database"""
        stats = {}
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Basic stats
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_articles,
                            COUNT(DISTINCT author) as unique_authors,
                            MIN(published_date) as earliest_date,
                            MAX(published_date) as latest_date,
                            AVG(word_count) as avg_word_count
                        FROM articles
                        WHERE published_date IS NOT NULL
                    """)
                    basic_stats = cursor.fetchone()
                    
                    stats.update(basic_stats)
                    
                    # Get top tags
                    cursor.execute("""
                        SELECT tags
                        FROM articles
                        WHERE tags IS NOT NULL 
                        LIMIT 1000
                    """)
                    tag_rows = cursor.fetchall()
                    
                    # Process tags
                    all_tags = []
                    for row in tag_rows:
                        if row['tags']:
                            try:
                                tags_value = row['tags']
                                
                                # PostgreSQL text[] arrays are returned as Python lists by psycopg2
                                if isinstance(tags_value, list):
                                    # Filter out NULL values
                                    valid_tags = [tag for tag in tags_value if tag and tag != 'NULL']
                                    all_tags.extend(valid_tags)
                                # If it's a string (shouldn't happen with text[] but just in case)
                                elif isinstance(tags_value, str):
                                    # Parse JSON string
                                    import json
                                    if tags_value.startswith('{'):
                                        # PostgreSQL array literal format
                                        tags_value = tags_value.strip('{}').split(',')
                                        valid_tags = [tag.strip('"') for tag in tags_value if tag and tag != 'NULL']
                                        all_tags.extend(valid_tags)
                                    elif tags_value.startswith('['):
                                        # JSON array format
                                        tags = json.loads(tags_value)
                                        if isinstance(tags, list):
                                            all_tags.extend(tags)
                            except Exception as e:
                                # Skip problematic entries
                                logger.debug(f"Could not parse tags: {e}")
                                continue
                    
                    # Count tag frequency
                    from collections import Counter
                    tag_counts = Counter(all_tags)
                    stats['top_tags'] = tag_counts.most_common(10)
                    stats['total_unique_tags'] = len(set(all_tags))
                    
                    # Articles by year
                    cursor.execute("""
                        SELECT 
                            EXTRACT(YEAR FROM published_date) as year,
                            COUNT(*) as count
                        FROM articles
                        WHERE published_date IS NOT NULL
                        GROUP BY year
                        ORDER BY year DESC
                    """)
                    stats['articles_by_year'] = cursor.fetchall()
                    
        except Exception as e:
            logger.error(f"Error getting article stats: {e}")
        
        return stats
    
    def _row_to_article(self, row: Dict[str, Any]) -> Article:
        """Convert a database row to an Article object"""
        return Article(
            id=row['id'],
            title=row['title'],
            url=row['url'],
            content=row.get('content'),
            excerpt=row.get('excerpt'),
            author=row.get('author'),
            published_date=row.get('published_date'),
            tags=row.get('tags'),
            word_count=row.get('word_count'),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            content_hash=row.get('content_hash')
        )

def test_database_manager():
    """Test the database manager"""
    print("\n" + "="*50)
    print("Testing Database Manager")
    print("="*50)
    
    db = DatabaseManager()
    
    # Test connection
    if not db.test_connection():
        print("Failed to connect to database!")
        return
    
    # Get some sample articles
    print("\n📚 Getting latest 3 articles...")
    latest = db.get_latest_articles(3)
    for article in latest:
        date_str = article.published_date.strftime("%Y-%m-%d") if article.published_date else "No date"
        print(f"  - {article.title[:50]}... ({date_str})")
    
    # Test keyword search
    print("\n🔍 Testing keyword search for 'DevOps'...")
    results = db.search_articles_by_keyword("DevOps")
    print(f"  Found {len(results)} articles")
    
    # Get stats
    print("\n📊 Database Statistics:")
    stats = db.get_article_stats()
    if stats:
        print(f"  Total articles: {stats.get('total_articles', 0)}")
        print(f"  Unique authors: {stats.get('unique_authors', 0)}")
        print(f"  Date range: {stats.get('earliest_date', 'N/A')} to {stats.get('latest_date', 'N/A')}")
        if 'top_tags' in stats and stats['top_tags']:
            print(f"  Top 3 tags: {', '.join([f'{tag}({count})' for tag, count in stats['top_tags'][:3]])}")
    
    print("\n✅ Database manager test complete!")

if __name__ == "__main__":
    test_database_manager()