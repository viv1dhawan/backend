# Backend/db_info/application_db.py - Updated for pyodbc
import uuid
from datetime import datetime

# Helper function to convert pyodbc.Row to a dictionary
def row_to_dict(row):
    if not row:
        return None
    return {column[0]: row[i] for i, column in enumerate(row.cursor_description)}

# Helper function to convert a list of pyodbc.Row objects to a list of dictionaries
def rows_to_dict_list(rows):
    if not rows:
        return []
    return [row_to_dict(row) for row in rows]

# --- Questions CRUD Operations ---

def get_all_questions_with_details(conn):
    """
    Retrieves all questions from the database, along with their comments
    and aggregated like/dislike counts.
    Questions are ordered by creation date (newest first).
    """
    cursor = conn.cursor()
    try:
        # Fetch all questions
        questions_query = "SELECT id, user_id, text, created_at FROM questions ORDER BY created_at DESC"
        cursor.execute(questions_query)
        question_rows = rows_to_dict_list(cursor.fetchall())
        if not question_rows:
            return []   # return a plain list

        question_ids = [q['id'] for q in question_rows]

        # Fetch all comments
        comments_query = "SELECT id, user_id, question_id, text, created_at FROM comments WHERE question_id IN ({})".format(
            ','.join(['?' for _ in question_ids])
        )
        cursor.execute(comments_query, question_ids)
        comment_rows = rows_to_dict_list(cursor.fetchall())

        # Fetch aggregated likes/dislikes
        interactions_query = """
        SELECT 
            question_id,
            SUM(CASE WHEN type='like' THEN 1 ELSE 0 END) AS likes_count,
            SUM(CASE WHEN type='dislike' THEN 1 ELSE 0 END) AS dislikes_count
        FROM question_interactions
        WHERE question_id IN ({})
        GROUP BY question_id
        """.format(','.join(['?' for _ in question_ids]))

        cursor.execute(interactions_query, question_ids)
        interaction_rows = rows_to_dict_list(cursor.fetchall())

        # Group comments
        comments_by_question = {}
        for comment in comment_rows:
            comments_by_question.setdefault(comment['question_id'], []).append(comment)

        # Map interactions
        interactions_by_question = {i['question_id']: i for i in interaction_rows}

        # Build final list (not wrapped in {"questions": ...})
        result = []
        for question in question_rows:
            q = {
                "id": question["id"],
                "text": question["text"],
                "created_at": question["created_at"],
                "comments": comments_by_question.get(question["id"], []),
                "likes_count": interactions_by_question.get(question["id"], {}).get("likes_count", 0),
                "dislikes_count": interactions_by_question.get(question["id"], {}).get("dislikes_count", 0),
            }
            result.append(q)

        return result  # list of questions, FastAPI will validate correctly

    finally:
        cursor.close()




def create_question_db(conn, user_id: int, text: str):
    """
    Creates a new question in the database.
    """
    cursor = conn.cursor()
    try:
        # Generate a unique ID for the question
        question_id = str(uuid.uuid4())
        query = "INSERT INTO questions (id, user_id, text) OUTPUT INSERTED.id, INSERTED.user_id, INSERTED.text, INSERTED.created_at VALUES (?, ?, ?);"
        cursor.execute(query, (question_id, user_id, text))
        new_question_row = cursor.fetchone()
        conn.commit()
        return row_to_dict(new_question_row)
    finally:
        cursor.close()

def get_question_by_id_db(conn, question_id: str):
    """
    Retrieves a single question by its ID.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT id, user_id, text, created_at FROM questions WHERE id = ?;"
        cursor.execute(query, (question_id,))
        return row_to_dict(cursor.fetchone())
    finally:
        cursor.close()

def get_comments_for_question_db(conn, question_id: str):
    """
    Retrieves all comments for a given question.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT id, user_id, question_id, text, created_at FROM comments WHERE question_id = ? ORDER BY created_at ASC;"
        cursor.execute(query, (question_id,))
        return rows_to_dict_list(cursor.fetchall())
    finally:
        cursor.close()

def add_comment_to_question_db(conn, question_id: str, user_id: int, text: str, interaction_type: str = None):
    """
    Adds a new comment to a question.
    """
    cursor = conn.cursor()
    try:
        # First, check if the question exists
        question = get_question_by_id_db(conn, question_id)
        if not question:
            return None

        created_at = datetime.now()
        query = "INSERT INTO comments (user_id, question_id, text, created_at) OUTPUT INSERTED.id, INSERTED.user_id, INSERTED.question_id, INSERTED.text, INSERTED.created_at VALUES (?, ?, ?, ?);"
        cursor.execute(query, (user_id, question_id, text, created_at))
        new_comment_row = cursor.fetchone()
        conn.commit()
        return row_to_dict(new_comment_row)
    finally:
        cursor.close()

def add_or_update_question_interaction_db(conn, user_id: int, question_id: str, interaction_type: str):
    """
    Adds or updates a user's interaction (like/dislike) for a question.
    Returns a dict of the inserted/updated row, or None if the question doesn't exist.
    """
    cursor = conn.cursor()
    try:
        # Validate question exists
        question = get_question_by_id_db(conn, question_id)
        if not question:
            return None

        now = datetime.now()

        # Try to update existing interaction
        update_sql = """
        UPDATE question_interactions
        SET type = ?, updated_at = ?
        OUTPUT INSERTED.id, INSERTED.user_id, INSERTED.question_id, INSERTED.type, INSERTED.updated_at
        WHERE user_id = ? AND question_id = ?;
        """
        cursor.execute(update_sql, (interaction_type, now, user_id, question_id))
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            result = dict(zip(columns, row))
            conn.commit()
            return result

        # Insert new interaction if none exists
        insert_sql = """
        INSERT INTO question_interactions (user_id, question_id, type, created_at)
        OUTPUT INSERTED.id, INSERTED.user_id, INSERTED.question_id, INSERTED.type, INSERTED.created_at
        VALUES (?, ?, ?, ?);
        """
        cursor.execute(insert_sql, (user_id, question_id, interaction_type, now))
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            result = dict(zip(columns, row))
            conn.commit()
            return result

        conn.commit()
        return None

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


# --- Researchers CRUD Operations ---

def get_researcher_by_id_db(conn, researcher_id: int):
    """
    Retrieves a single researcher by ID.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT id, title, authors, user_id, profile, publication_date, url, created_at, updated_at FROM researchers WHERE id = ?;"
        cursor.execute(query, (researcher_id,))
        return row_to_dict(cursor.fetchone())
    finally:
        cursor.close()

def get_researcher_by_id_and_user_id_db(conn, researcher_id: int, user_id: int):
    """
    Retrieves a single researcher by ID and user ID.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT id, title, authors, user_id, profile, publication_date, url, created_at, updated_at FROM researchers WHERE id = ? AND user_id = ?;"
        cursor.execute(query, (researcher_id, user_id))
        return row_to_dict(cursor.fetchone())
    finally:
        cursor.close()


def add_researcher_db(conn, researcher_data: dict, user_id: int):
    """
    Adds a new researcher entry to the database.
    """
    cursor = conn.cursor()
    try:
        query = """
        INSERT INTO researchers (title, authors, user_id, profile, publication_date, url)
        OUTPUT INSERTED.id, INSERTED.title, INSERTED.authors, INSERTED.user_id, INSERTED.profile, INSERTED.publication_date, INSERTED.url, INSERTED.created_at, INSERTED.updated_at
        VALUES (?, ?, ?, ?, ?, ?);
        """
        params = (
            researcher_data["title"],
            researcher_data["authors"],
            user_id, # Use the authenticated user's ID
            researcher_data["profile"],
            researcher_data["publication_date"],
            researcher_data["url"]
        )
        cursor.execute(query, params)
        new_researcher_row = cursor.fetchone()
        conn.commit()
        return row_to_dict(new_researcher_row)
    finally:
        cursor.close()

def update_researcher_db(conn, researcher_data: dict, user_id: int):
    """
    Updates an existing researcher entry, ensuring ownership.
    """
    cursor = conn.cursor()
    try:
        researcher_id = researcher_data.get("researcher_id")
        existing_researcher = get_researcher_by_id_and_user_id_db(conn, researcher_id, user_id)
        if not existing_researcher:
            return None # Not found or not owned by user
            
        set_clauses = []
        params = []
        
        # Update logic based on provided fields
        if "title" in researcher_data and researcher_data["title"] is not None:
            set_clauses.append("title = ?")
            params.append(researcher_data["title"])
        if "authors" in researcher_data and researcher_data["authors"] is not None:
            set_clauses.append("authors = ?")
            params.append(researcher_data["authors"])
        if "profile" in researcher_data and researcher_data["profile"] is not None:
            set_clauses.append("profile = ?")
            params.append(researcher_data["profile"])
        if "publication_date" in researcher_data and researcher_data["publication_date"] is not None:
            set_clauses.append("publication_date = ?")
            params.append(researcher_data["publication_date"])
        if "url" in researcher_data and researcher_data["url"] is not None:
            set_clauses.append("url = ?")
            params.append(researcher_data["url"])

        if not set_clauses:
            return existing_researcher
        
        set_clauses.append("updated_at = GETDATE()")
        
        query = f"UPDATE researchers SET {', '.join(set_clauses)} WHERE id = ? AND user_id = ?"
        params.append(researcher_id)
        params.append(user_id)
        
        cursor.execute(query, params)
        conn.commit()
        return get_researcher_by_id_and_user_id_db(conn, researcher_id, user_id)
    finally:
        cursor.close()

def delete_researcher_db(conn, researcher_id: int, user_id: int):
    """
    Deletes a researcher entry from the database, ensuring ownership.
    """
    cursor = conn.cursor()
    try:
        query = "DELETE FROM researchers WHERE id = ? AND user_id = ?;"
        cursor.execute(query, (researcher_id, user_id))
        conn.commit()
        return cursor.rowcount > 0 # Returns True if a row was deleted, False otherwise
    finally:
        cursor.close()

def get_all_researchers_db(conn):
    """
    Retrieves all researcher entries.
    """
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM researchers ORDER BY created_at DESC;"
        cursor.execute(query)
        return rows_to_dict_list(cursor.fetchall())
    finally:
        cursor.close()
