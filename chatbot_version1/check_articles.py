#!/usr/bin/env python3
"""
Script to investigate missing article
"""

import psycopg2
from config import get_postgres_connection

def check_articles():
    """Check article counts and find missing ones"""
    conn = get_postgres_connection()
    if not conn:
        print("Could not connect to PostgreSQL")
        return
    
    try:
        cursor = conn.cursor()
        
        # Total count of all articles
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_count = cursor.fetchone()[0]
        print(f"Total articles in database: {total_count}")
        
        # Count of articles with content
        cursor.execute("SELECT COUNT(*) FROM articles WHERE content IS NOT NULL AND content != ''")
        content_count = cursor.fetchone()[0]
        print(f"Articles with non-empty content: {content_count}")
        
        # Count of articles without content
        cursor.execute("SELECT COUNT(*) FROM articles WHERE content IS NULL OR content = ''")
        no_content_count = cursor.fetchone()[0]
        print(f"Articles with empty/null content: {no_content_count}")
        
        if no_content_count > 0:
            print("\nArticles with missing content:")
            cursor.execute("""
                SELECT id, title, url, content 
                FROM articles 
                WHERE content IS NULL OR content = ''
                ORDER BY id
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                article_id, title, url, content = row
                content_preview = content[:50] + "..." if content else "NULL/Empty"
                print(f"  ID {article_id}: {title} - Content: {content_preview}")
        
        # Check for other potential issues
        cursor.execute("SELECT COUNT(*) FROM articles WHERE title IS NULL OR title = ''")
        no_title_count = cursor.fetchone()[0]
        print(f"\nArticles with empty/null titles: {no_title_count}")
        
    except Exception as e:
        print(f"Error checking articles: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_articles()