# Backend/db_info/application_db.py
import uuid
import pandas as pd
import io
from datetime import datetime
from typing import List, Optional, Dict, Any
import aiomysql # Import aiomysql

# No longer importing SQLAlchemy models here
# from Schema.application_schema import CommentCreate, CommentResponse, EarthquakeQuery, QuestionCreate, QuestionResponse, LikeDislikeType, QuestionInteractionResponse, Researcher

# You will need to define these enums or their string values directly since Pydantic schemas are no longer strictly mapping to SQLAlchemy models for database operations
class LikeDislikeType:
    LIKE = "like"
    DISLIKE = "dislike"


# --- Gravity Data Operations ---

async def load_gravity_data_from_csv(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, csv_contents: bytes) -> int:
    """
    Loads gravity data from a CSV file into the database.
    Assumes CSV has 'latitude', 'longitude', 'elevation', 'gravity' columns.
    Clears existing gravity data before inserting new data.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        csv_contents (bytes): The content of the CSV file.
    Returns:
        int: The number of rows inserted.
    """
    df = pd.read_csv(io.BytesIO(csv_contents))

    # Convert DataFrame column names to lowercase
    df.columns = df.columns.str.lower()

    # Validate required columns
    required_columns = ['latitude', 'longitude', 'elevation', 'gravity']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain the following columns: {', '.join(required_columns)}")

    # Clear existing data before bulk insert
    await clear_gravity_data(conn, cursor)

    records_to_insert = df[required_columns].to_dict(orient="records")
    
    if records_to_insert:
        # Prepare for bulk insert
        insert_query = """
        INSERT INTO gravity_data (latitude, longitude, elevation, gravity, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        values_to_insert = []
        for record in records_to_insert:
            values_to_insert.append((
                record['latitude'],
                record['longitude'],
                record['elevation'],
                record['gravity'],
                datetime.now()
            ))
        
        await cursor.executemany(insert_query, values_to_insert)
        await conn.commit()

    return len(records_to_insert)

async def get_gravity_data(conn: aiomysql.Connection, cursor: aiomysql.DictCursor) -> pd.DataFrame:
    """
    Retrieves all gravity data from the database and returns it as a Pandas DataFrame.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
    Returns:
        pd.DataFrame: A DataFrame containing all gravity data.
    """
    query = "SELECT id, latitude, longitude, elevation, gravity, bouguer, cluster, anomaly, distance_km FROM gravity_data"
    await cursor.execute(query)
    records = await cursor.fetchall()

    if not records:
        # Return an empty DataFrame with all expected columns to maintain consistent schema
        return pd.DataFrame(columns=['id', 'latitude', 'longitude', 'elevation', 'gravity', 'bouguer', 'cluster', 'anomaly', 'distance_km'])

    df = pd.DataFrame(records)
    return df

async def clear_gravity_data(conn: aiomysql.Connection, cursor: aiomysql.DictCursor) -> None:
    """
    Clears all gravity data from the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
    """
    query = "DELETE FROM gravity_data"
    await cursor.execute(query)
    await conn.commit()

async def update_gravity_data(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, df: pd.DataFrame) -> None:
    """
    Updates existing gravity data in the database based on the DataFrame.
    This function assumes the DataFrame contains an 'id' column for existing records.
    For calculated fields (bouguer, cluster, anomaly, distance_km), it updates them.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        df (pd.DataFrame): The DataFrame containing gravity data to update.
    """
    if 'id' not in df.columns:
        print("Warning: 'id' column not found in DataFrame for update. Please ensure data has IDs.")
        return # Cannot update without IDs

    update_statements = []
    update_params = []

    for index, row in df.iterrows():
        record_id = row['id']
        update_values = {}
        
        for col in ['latitude', 'longitude', 'elevation', 'gravity', 'bouguer', 'cluster', 'anomaly', 'distance_km']:
            if col in row and pd.notna(row[col]):
                update_values[col] = row[col]

        if update_values:
            set_clauses = []
            params_for_row = []
            for field, value in update_values.items():
                set_clauses.append(f"{field} = %s")
                params_for_row.append(value)
            
            params_for_row.append(record_id)
            update_statements.append(f"UPDATE gravity_data SET {', '.join(set_clauses)} WHERE id = %s;")
            update_params.append(params_for_row)

    if update_statements:
        # aiomysql's executemany for multiple UPDATE statements is not straightforward
        # It's safer to execute them one by one or reconstruct the full query if possible.
        # For simplicity and to match user's request for "cursor.execute", we'll loop.
        for i, stmt in enumerate(update_statements):
            await cursor.execute(stmt, update_params[i])
        await conn.commit()


# --- Earthquake Data Operations ---
async def get_earthquakes(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetches earthquake data from the database based on the provided query parameters.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        query_params (Dict[str, Any]): The query parameters for filtering earthquakes.
    Returns:
        List[Dict[str, Any]]: A list of earthquake data.
    """
    sql_query = "SELECT id, time, latitude, longitude, depth, mag, place, updated_at FROM earthquakes WHERE time BETWEEN %s AND %s"
    params = [query_params["start_date"], query_params["end_date"]]

    if query_params.get("min_mag") is not None:
        sql_query += " AND mag >= %s"
        params.append(query_params["min_mag"])
    if query_params.get("max_mag") is not None:
        sql_query += " AND mag <= %s"
        params.append(query_params["max_mag"])
    if query_params.get("min_depth") is not None:
        sql_query += " AND depth >= %s"
        params.append(query_params["min_depth"])
    if query_params.get("max_depth") is not None:
        sql_query += " AND depth <= %s"
        params.append(query_params["max_depth"])

    await cursor.execute(sql_query, tuple(params))
    records = await cursor.fetchall()
    return records


# --- Questions CRUD Operations ---

async def get_all_questions_with_details(conn: aiomysql.Connection, cursor: aiomysql.DictCursor) -> List[Dict[str, Any]]:
    """
    Retrieves all questions from the database, along with their comments
    and aggregated like/dislike counts.
    Questions are ordered by creation date (newest first).
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
    Returns:
        List[Dict[str, Any]]: A list of questions with their details.
    """
    # 1. Fetch all questions
    questions_query = "SELECT id, text, created_at FROM questions ORDER BY created_at DESC"
    await cursor.execute(questions_query)
    question_rows = await cursor.fetchall()

    # 2. Fetch all comments and group by question_id
    comments_query = "SELECT id, text, created_at, question_id FROM comments ORDER BY created_at DESC"
    await cursor.execute(comments_query)
    comment_rows = await cursor.fetchall()

    comments_by_question: Dict[str, List[Dict[str, Any]]] = {}
    for row in comment_rows:
        comments_by_question.setdefault(row['question_id'], []).append({
            "id": row['id'],
            "text": row['text'],
            "created_at": row['created_at']
        })

    # 3. Fetch all like/dislike counts and group by question_id
    likes_dislikes_query = f"""
    SELECT
        question_id,
        SUM(CASE WHEN type = '{LikeDislikeType.LIKE}' THEN 1 ELSE 0 END) AS likes_count,
        SUM(CASE WHEN type = '{LikeDislikeType.DISLIKE}' THEN 1 ELSE 0 END) AS dislikes_count
    FROM question_likes
    GROUP BY question_likes.question_id
    """
    await cursor.execute(likes_dislikes_query)
    likes_dislikes_rows = await cursor.fetchall()

    counts_by_question: Dict[str, Dict[str, int]] = {
        row['question_id']: {"likes_count": row['likes_count'], "dislikes_count": row['dislikes_count']}
        for row in likes_dislikes_rows
    }

    # 4. Assemble the final list of dictionaries
    response_questions: List[Dict[str, Any]] = []
    for q_row in question_rows:
        question_id = q_row['id']
        question_data = {
            "id": question_id,
            "text": q_row['text'],
            "created_at": q_row['created_at'],
            "comments": comments_by_question.get(question_id, []),
            "likes_count": counts_by_question.get(question_id, {}).get("likes_count", 0),
            "dislikes_count": counts_by_question.get(question_id, {}).get("dislikes_count", 0)
        }
        response_questions.append(question_data)

    return response_questions

async def create_question_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, text: str, user_id: int) -> Dict[str, Any]:
    """
    Creates a new question in the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        text (str): The text content of the question.
        user_id (int): The ID of the user asking the question.
    Returns:
        Dict[str, Any]: The created question's details.
    """
    new_question_id = str(uuid.uuid4()) # Generate UUID for question ID
    query = """
    INSERT INTO questions (id, text, user_id, created_at)
    VALUES (%s, %s, %s, %s)
    """
    params = (new_question_id, text, user_id, datetime.now())
    await cursor.execute(query, params)
    await conn.commit() # Commit after insert

    # Fetch the newly created question to return its details
    return await get_question_by_id_with_details(conn, cursor, new_question_id)

async def get_question_by_id_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, question_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single question by its ID.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        question_id (str): The ID of the question.
    Returns:
        The question record or None if not found.
    """
    query = "SELECT id, text, created_at, user_id FROM questions WHERE id = %s"
    await cursor.execute(query, (question_id,))
    return await cursor.fetchone()

async def get_question_by_id_with_details(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, question_id: str) -> Dict[str, Any]:
    """
    Retrieves a single question by its ID, along with its comments
    and aggregated like/dislike counts.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        question_id (str): The ID of the question.
    Returns:
        Dict[str, Any]: The question with its full details.
    Raises:
        ValueError: If the question with the given ID is not found.
    """
    # 1. Fetch the question
    question_query = "SELECT id, text, created_at FROM questions WHERE id = %s"
    await cursor.execute(question_query, (question_id,))
    q_row = await cursor.fetchone()

    if not q_row:
        raise ValueError(f"Question with ID {question_id} not found.")

    # 2. Fetch comments for this question
    comments_query = "SELECT id, text, created_at FROM comments WHERE question_id = %s ORDER BY created_at DESC"
    await cursor.execute(comments_query, (question_id,))
    comment_rows = await cursor.fetchall()

    comments_list = [
        {"id": row['id'], "text": row['text'], "created_at": row['created_at']}
        for row in comment_rows
    ]

    # 3. Fetch like/dislike counts for this question
    likes_dislikes_query = f"""
    SELECT
        SUM(CASE WHEN type = '{LikeDislikeType.LIKE}' THEN 1 ELSE 0 END) AS likes_count,
        SUM(CASE WHEN type = '{LikeDislikeType.DISLIKE}' THEN 1 ELSE 0 END) AS dislikes_count
    FROM question_likes
    WHERE question_id = %s
    """
    await cursor.execute(likes_dislikes_query, (question_id,))
    counts_row = await cursor.fetchone()

    likes_count = counts_row['likes_count'] if counts_row and counts_row['likes_count'] else 0
    dislikes_count = counts_row['dislikes_count'] if counts_row and counts_row['dislikes_count'] else 0

    return {
        "id": q_row['id'],
        "text": q_row['text'],
        "created_at": q_row['created_at'],
        "comments": comments_list,
        "likes_count": likes_count,
        "dislikes_count": dislikes_count
    }

async def create_comment_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, question_id: str, text: str) -> Dict[str, Any]:
    """
    Creates a new comment for a given question.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        question_id (str): The ID of the question to comment on.
        text (str): The text content of the comment.
    Returns:
        Dict[str, Any]: The created comment's details.
    Raises:
        ValueError: If the newly created comment cannot be retrieved.
    """
    new_comment_id = str(uuid.uuid4()) # Generate UUID for comment ID
    query = """
    INSERT INTO comments (id, question_id, text, created_at)
    VALUES (%s, %s, %s, %s)
    """
    params = (new_comment_id, question_id, text, datetime.now())
    await cursor.execute(query, params)
    # No commit here, handled by caller in route or service layer
    # As per previous SQLAlchemy versions where commit was sometimes delayed

    result_query = "SELECT id, question_id, text, created_at FROM comments WHERE id = %s"
    await cursor.execute(result_query, (new_comment_id,))
    created_comment = await cursor.fetchone()

    if created_comment:
        return created_comment
    raise ValueError("Failed to retrieve the newly created comment.")

async def create_question_interaction_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, question_id: str, user_id: int, interaction_type: str) -> Dict[str, Any]:
    """
    Records a new like or dislike interaction for a question.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        question_id (str): The ID of the question being interacted with.
        user_id (int): The ID of the user performing the interaction.
        interaction_type (str): The type of interaction ('like' or 'dislike').
    Returns:
        Dict[str, Any]: The created interaction's details.
    Raises:
        ValueError: If the newly created interaction cannot be retrieved.
    """
    new_interaction_id = str(uuid.uuid4())
    query = """
    INSERT INTO question_likes (id, question_id, user_id, type, created_at)
    VALUES (%s, %s, %s, %s, %s)
    """
    params = (new_interaction_id, question_id, user_id, interaction_type, datetime.now())
    await cursor.execute(query, params)
    # Commit handled by caller

    result_query = "SELECT id, question_id, user_id, type, created_at FROM question_likes WHERE id = %s"
    await cursor.execute(result_query, (new_interaction_id,))
    created_interaction = await cursor.fetchone()
    if created_interaction:
        return created_interaction
    raise ValueError("Failed to retrieve the newly created interaction.")

async def get_user_question_interaction_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, user_id: int, question_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user's existing like/dislike interaction for a specific question.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        user_id (int): The ID of the user.
        question_id (str): The ID of the question.
    Returns:
        Optional[Dict[str, Any]]: The interaction details or None if not found.
    """
    query = "SELECT id, question_id, user_id, type, created_at FROM question_likes WHERE user_id = %s AND question_id = %s"
    await cursor.execute(query, (user_id, question_id))
    row = await cursor.fetchone()
    return row

async def update_question_interaction_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, interaction_id: str, new_type: str) -> Dict[str, Any]:
    """
    Updates an existing like/dislike interaction.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        interaction_id (str): The ID of the interaction to update.
        new_type (str): The new type of interaction.
    Returns:
        Dict[str, Any]: The updated interaction details.
    Raises:
        ValueError: If the updated interaction cannot be retrieved.
    """
    current_time = datetime.now()
    query = "UPDATE question_likes SET type = %s, created_at = %s WHERE id = %s"
    params = (new_type, current_time, interaction_id)
    await cursor.execute(query, params)
    # Commit handled by caller

    # Fetch the updated row to return
    result_query = "SELECT id, question_id, user_id, type, created_at FROM question_likes WHERE id = %s"
    await cursor.execute(result_query, (interaction_id,))
    updated_row = await cursor.fetchone()
    if updated_row:
        return updated_row
    raise ValueError("Updated interaction not found, something went wrong during update and fetch.")


async def delete_question_interaction_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, interaction_id: str):
    """
    Deletes a like/dislike interaction.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        interaction_id (str): The ID of the interaction to delete.
    """
    query = "DELETE FROM question_likes WHERE id = %s"
    await cursor.execute(query, (interaction_id,))
    # Commit handled by caller


# --- User & Existence Checks (Helper Functions) ---

async def user_exists_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, user_id: int) -> bool:
    """
    Checks if a user with the given ID exists.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        user_id (int): The ID of the user to check.
    Returns:
        bool: True if the user exists, False otherwise.
    """
    query = "SELECT id FROM users WHERE id = %s"
    await cursor.execute(query, (user_id,))
    result = await cursor.fetchone()
    return result is not None

# --- Researcher CRUD Operations ---

async def create_researcher_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, researcher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new researcher entry in the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        researcher_data (Dict[str, Any]): Dictionary containing researcher details.
    Returns:
        Dict[str, Any]: The created researcher's data.
    """
    # Assuming 'id' for researchers is AUTO_INCREMENT, so no need to generate UUID for it if the table schema is INT AUTO_INCREMENT
    # If it was string UUID in the old schema, adjust this to generate UUID or use AUTO_INCREMENT for id.
    # The provided models.py for 'researchers' had 'id' as 'sqlalchemy.Integer, primary_key=True, autoincrement=True)'
    # so we will rely on auto-increment.

    query = """
    INSERT INTO researchers (title, authors, profile, publication_date, url, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        researcher_data["title"],
        researcher_data["authors"],
        researcher_data["profile"],
        researcher_data["publication_date"], # Stored as string
        researcher_data["url"],
        datetime.now(),
        datetime.now()
    )
    await cursor.execute(query, params)
    await conn.commit()

    # Get the last inserted ID
    await cursor.execute("SELECT LAST_INSERT_ID() as id")
    result = await cursor.fetchone()
    last_record_id = result['id'] if result else None
    
    if last_record_id:
        return await get_researcher_by_id_db(conn, cursor, last_record_id)
    return None

async def get_all_researchers_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor) -> List[Dict[str, Any]]:
    """
    Retrieves all researcher entries from the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
    Returns:
        List[Dict[str, Any]]: A list of all researcher data.
    """
    query = "SELECT id, title, authors, profile, publication_date, url, created_at, updated_at FROM researchers ORDER BY created_at DESC"
    await cursor.execute(query)
    return await cursor.fetchall()

async def get_researcher_by_id_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, researcher_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single researcher entry by its ID.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        researcher_id (str): The ID of the researcher.
    Returns:
        Optional[Dict[str, Any]]: The researcher's data or None if not found.
    """
    # Note: If researcher_id is INT in DB, ensure it's passed as INT from FastAPI
    query = "SELECT id, title, authors, profile, publication_date, url, created_at, updated_at FROM researchers WHERE id = %s"
    await cursor.execute(query, (researcher_id,))
    return await cursor.fetchone()

async def update_researcher_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, researcher_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Updates an existing researcher entry in the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        researcher_id (str): The ID of the researcher to update.
        update_data (Dict[str, Any]): Dictionary of fields to update.
    Returns:
        Optional[Dict[str, Any]]: The updated researcher's data or None if update failed.
    """
    update_data["updated_at"] = datetime.now()
    
    set_clauses = []
    params = []
    for field, value in update_data.items():
        set_clauses.append(f"{field} = %s")
        params.append(value)
    
    if not set_clauses:
        return await get_researcher_by_id_db(conn, cursor, researcher_id) # Nothing to update
    
    params.append(researcher_id) # Add researcher_id for the WHERE clause

    query = f"UPDATE researchers SET {', '.join(set_clauses)} WHERE id = %s"
    await cursor.execute(query, params)
    await conn.commit()
    return await get_researcher_by_id_db(conn, cursor, researcher_id)

async def delete_researcher_db(conn: aiomysql.Connection, cursor: aiomysql.DictCursor, researcher_id: str) -> bool:
    """
    Deletes a researcher entry from the database.
    Args:
        conn: The aiomysql connection.
        cursor: The aiomysql dictionary cursor.
        researcher_id (str): The ID of the researcher to delete.
    Returns:
        bool: True if a row was deleted, False otherwise.
    """
    query = "DELETE FROM researchers WHERE id = %s"
    await cursor.execute(query, (researcher_id,))
    await conn.commit()
    return cursor.rowcount > 0 # Return True if a row was deleted
