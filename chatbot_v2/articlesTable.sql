-- Run this in Adminer

CREATE TABLE articles (
    id SERIAL PRIMARY KEY,              -- Auto-incrementing unique identifier
    title VARCHAR(500) NOT NULL,        -- Article headline (max 500 chars)
    url VARCHAR(1000) UNIQUE NOT NULL,  -- Article URL (unique, required)
    content TEXT,                       -- Full article text (unlimited length)
    excerpt TEXT,                       -- Short summary
    author VARCHAR(200),                -- Author name (max 200 chars)
    published_date TIMESTAMP,           -- When article was published
    tags TEXT[],                        -- Array of categortopic tags
    word_count INTEGER,                 -- Num words in article
    created_at TIMESTAMP DEFAULT NOW(), -- When record was added to DB
    updated_at TIMESTAMP DEFAULT NOW()  -- When record was last modified
    content_hash VARCHAR(64) NOT NULL  -- Hash of content for change detection
);

--  For URL searches -  "does this URL already exist?"
CREATE INDEX idx_articles_url ON articles(url);

-- For published date searches - "show recent articles"
CREATE INDEX idx_articles_published ON articles(published_date);

-- For tag searches - "find articles tagged with 'React'"
CREATE INDEX idx_articles_tags ON articles USING GIN(tags);

-- for content_hash


-- Deal with canonical URLs (redirects of older links to new versions)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS url_canonical VARCHAR(1000);
CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_canonical ON articles(url_canonical);


-- ADD VECCTOR EMBEDDINGS - to add vector search capabilities
CREATE EXTENTSION IF NOT EXISTS vector;

ALTER TABLE articles ADD COLUMN embedding vector(1536);

CREATE INDEX ON articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);



