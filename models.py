# models.py - MySQL table schemas

# Users table
create_users_table = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    is_verified BOOLEAN DEFAULT 0 NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
"""

# Password reset tokens
create_password_reset_tokens_table = """
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    token VARCHAR(255) NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL
);
"""

# Email verification tokens
create_email_verification_tokens_table = """
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    token VARCHAR(255) NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL
);
"""

# Questions
create_questions_table = """
CREATE TABLE IF NOT EXISTS questions (
    id CHAR(36) PRIMARY KEY,
    user_id INT,
    text TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

# Comments
create_comments_table = """
CREATE TABLE IF NOT EXISTS comments (
    id CHAR(36) PRIMARY KEY,
    user_id INT,
    question_id CHAR(36),
    text TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);
"""

# Question interactions
create_question_interactions_table = """
CREATE TABLE IF NOT EXISTS question_interactions (
    id CHAR(36) PRIMARY KEY,
    user_id INT,
    question_id CHAR(36),
    type VARCHAR(10) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, question_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);
"""

# Researchers
create_researchers_table = """
CREATE TABLE IF NOT EXISTS researchers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    authors VARCHAR(500) NOT NULL,
    user_id INT,
    profile VARCHAR(500) NOT NULL,
    publication_date VARCHAR(50) NOT NULL,
    url VARCHAR(1000) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

# List of tables to create in order
TABLES = [
    create_users_table,
    create_password_reset_tokens_table,
    create_email_verification_tokens_table,
    create_questions_table,
    create_comments_table,
    create_question_interactions_table,
    create_researchers_table,
]
