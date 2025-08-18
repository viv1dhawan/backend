# Backend/db_info/gravity_db.py
import uuid
import pandas as pd
import io
import sqlalchemy
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession # Import AsyncSession
from sqlalchemy import select, desc, func, and_, insert, delete, update # Import necessary SQLAlchemy functions

from Schema.application_schema import CommentCreate, CommentResponse, EarthquakeQuery, QuestionCreate, QuestionResponse, LikeDislikeType, QuestionInteractionResponse, Researcher # Import CommentCreate and LikeDislikeType
from models import gravity_data, earthquakes, questions, comments, question_likes, users, researchers # Import all models used

# --- Gravity Data Operations ---

async def load_gravity_data_from_csv(session: AsyncSession, csv_contents: bytes) -> int:
    """
    Loads gravity data from a CSV file into the database.
    Assumes CSV has 'latitude', 'longitude', 'elevation', 'gravity' columns.
    Clears existing gravity data before inserting new data.
    Args:
        session (AsyncSession): The database session.
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

    # Convert DataFrame to a list of dictionaries for insertion
    records_to_insert = df[required_columns].to_dict(orient="records")

    # Clear existing data before bulk insert
    await clear_gravity_data(session)

    # Bulk insert the new data
    if records_to_insert:
        query = gravity_data.insert()
        await session.execute(query, records_to_insert) # execute_many is used by passing a list of dicts to execute()
        await session.commit()
    return len(records_to_insert)

async def get_gravity_data(session: AsyncSession) -> pd.DataFrame:
    """
    Retrieves all gravity data from the database and returns it as a Pandas DataFrame.
    Args:
        session (AsyncSession): The database session.
    Returns:
        pd.DataFrame: A DataFrame containing all gravity data.
    """
    query = gravity_data.select()
    result = await session.execute(query)
    records = result.fetchall()
    if not records:
        # Return an empty DataFrame with all expected columns to maintain consistent schema
        return pd.DataFrame(columns=['id', 'latitude', 'longitude', 'elevation', 'gravity', 'bouguer', 'cluster', 'anomaly', 'distance_km'])

    # Convert records to a list of dictionaries and then to DataFrame
    df = pd.DataFrame([dict(record) for record in records])
    return df

async def clear_gravity_data(session: AsyncSession) -> None:
    """
    Clears all gravity data from the database.
    Args:
        session (AsyncSession): The database session.
    """
    query = gravity_data.delete()
    await session.execute(query)
    await session.commit()

async def update_gravity_data(session: AsyncSession, df: pd.DataFrame) -> None:
    """
    Updates existing gravity data in the database based on the DataFrame.
    This function assumes the DataFrame contains an 'id' column for existing records.
    For calculated fields (bouguer, cluster, anomaly, distance_km), it updates them.
    If no 'id' column is present, it clears and re-inserts all data.
    Args:
        session (AsyncSession): The database session.
        df (pd.DataFrame): The DataFrame containing gravity data to update.
    """
    if 'id' not in df.columns:
        # If no ID, clear and re-insert (less efficient but ensures data is consistent)
        print("Warning: 'id' column not found in DataFrame for update. Clearing and re-inserting gravity data.")
        await clear_gravity_data(session)
        records_to_insert = df.to_dict(orient="records")
        if records_to_insert:
            query = gravity_data.insert()
            await session.execute(query, records_to_insert)
            await session.commit()
    else:
        # Update existing records based on ID
        for index, row in df.iterrows():
            record_id = row['id']
            # Only include columns present in the DataFrame for update
            update_values = {
                col: row[col] for col in ['latitude', 'longitude', 'elevation', 'gravity', 'bouguer', 'cluster', 'anomaly', 'distance_km']
                if col in row and pd.notna(row[col]) # Check for existence and not NaN
            }

            if update_values: # Only execute if there are values to update
                query = gravity_data.update().where(gravity_data.c.id == record_id).values(**update_values)
                await session.execute(query)
        await session.commit()

# --- Earthquake Data Operations ---
async def get_earthquakes(session: AsyncSession, query: EarthquakeQuery) -> List[Dict[str, Any]]:
    """
    Fetches earthquake data from the database based on the provided query parameters.
    Args:
        session (AsyncSession): The database session.
        query (EarthquakeQuery): The query parameters for filtering earthquakes.
    Returns:
        List[Dict[str, Any]]: A list of earthquake data.
    """
    sql_query = earthquakes.select().where(
        earthquakes.c.time >= query.start_date,
        earthquakes.c.time <= query.end_date
    )

    if query.min_mag is not None:
        sql_query = sql_query.where(earthquakes.c.mag >= query.min_mag)
    if query.max_mag is not None:
        sql_query = sql_query.where(earthquakes.c.mag <= query.max_mag)
    if query.min_depth is not None:
        sql_query = sql_query.where(earthquakes.c.depth <= query.min_depth)
    if query.max_depth is not None:
        sql_query = sql_query.where(earthquakes.c.depth <= query.max_depth)

    result = await session.execute(sql_query)
    records = result.fetchall()

    return [dict(record) for record in records]


# --- Questions CRUD Operations ---

async def get_all_questions_with_details(session: AsyncSession) -> List[QuestionResponse]:
    """
    Retrieves all questions from the database, along with their comments
    and aggregated like/dislike counts.
    Questions are ordered by creation date (newest first).
    Args:
        session (AsyncSession): The database session.
    Returns:
        List[QuestionResponse]: A list of questions with their details.
    """
    # 1. Fetch all questions
    q_stmt = select(
        questions.c.id,
        questions.c.text,
        questions.c.created_at
    ).order_by(desc(questions.c.created_at))
    question_result = await session.execute(q_stmt)
    question_rows = question_result.fetchall()

    # 2. Fetch all comments and group by question_id
    c_stmt = select(
        comments.c.id,
        comments.c.text,
        comments.c.created_at,
        comments.c.question_id
    ).order_by(desc(comments.c.created_at))
    comment_result = await session.execute(c_stmt)
    comment_rows = comment_result.fetchall()

    comments_by_question: Dict[str, List[CommentResponse]] = {}
    for row in comment_rows:
        comment_data = CommentResponse(
            id=row.id,
            text=row.text,
            created_at=row.created_at
        )
        comments_by_question.setdefault(row.question_id, []).append(comment_data)

    # 3. Fetch all like/dislike counts and group by question_id
    likes_dislikes_stmt = select(
        question_likes.c.question_id,
        func.sum(sqlalchemy.case((question_likes.c.type == LikeDislikeType.LIKE.value, 1), else_=0)).label("likes_count"), # Use .value for enum
        func.sum(sqlalchemy.case((question_likes.c.type == LikeDislikeType.DISLIKE.value, 1), else_=0)).label("dislikes_count") # Use .value for enum
    ).group_by(question_likes.c.question_id)
    likes_dislikes_result = await session.execute(likes_dislikes_stmt)
    likes_dislikes_rows = likes_dislikes_result.fetchall()

    counts_by_question: Dict[str, Dict[str, int]] = {
        row.question_id: {"likes_count": row.likes_count, "dislikes_count": row.dislikes_count}
        for row in likes_dislikes_rows
    }

    # 4. Assemble the final list of QuestionResponse objects
    response_questions: List[QuestionResponse] = []
    for q_row in question_rows:
        question_id = q_row.id
        question_data = {
            "id": question_id,
            "text": q_row.text,
            "created_at": q_row.created_at,
            "comments": comments_by_question.get(question_id, []),
            "likes_count": counts_by_question.get(question_id, {}).get("likes_count", 0),
            "dislikes_count": counts_by_question.get(question_id, {}).get("dislikes_count", 0)
        }
        response_questions.append(QuestionResponse(**question_data))

    return response_questions

async def create_question_db(session: AsyncSession, text: str, user_id: int) -> QuestionResponse:
    """
    Creates a new question in the database.
    Args:
        session (AsyncSession): The database session.
        text (str): The text content of the question.
        user_id (int): The ID of the user asking the question.
    Returns:
        QuestionResponse: The created question's details.
    """
    new_question_id = str(uuid.uuid4()) # Generate UUID for question ID
    query = questions.insert().values(
        id=new_question_id,
        text=text,
        user_id=user_id, # Assign the user_id here
        created_at=datetime.now()
    )
    await session.execute(query)
    await session.commit() # Commit after insert

    # Fetch the newly created question to return its details
    result = await session.execute(select(questions).where(questions.c.id == new_question_id))
    created_question = result.first()
    if created_question:
        # For a new question, comments and likes/dislikes are empty
        return QuestionResponse(
            id=created_question.id,
            text=created_question.text,
            created_at=created_question.created_at,
            comments=[],
            likes_count=0,
            dislikes_count=0
        )
    raise ValueError("Failed to retrieve the newly created question.")

async def get_question_by_id_db(session: AsyncSession, question_id: str):
    """
    Retrieves a single question by its ID.
    Args:
        session (AsyncSession): The database session.
        question_id (str): The ID of the question.
    Returns:
        The question record or None if not found.
    """
    stmt = select(questions).where(questions.c.id == question_id)
    result = await session.execute(stmt)
    return result.first()

async def get_question_by_id_with_details(session: AsyncSession, question_id: str) -> QuestionResponse:
    """
    Retrieves a single question by its ID, along with its comments
    and aggregated like/dislike counts.
    Args:
        session (AsyncSession): The database session.
        question_id (str): The ID of the question.
    Returns:
        QuestionResponse: The question with its full details.
    Raises:
        ValueError: If the question with the given ID is not found.
    """
    # 1. Fetch the question
    q_stmt = select(
        questions.c.id,
        questions.c.text,
        questions.c.created_at
    ).where(questions.c.id == question_id)
    question_result = await session.execute(q_stmt)
    q_row = question_result.first()

    if not q_row:
        raise ValueError(f"Question with ID {question_id} not found.")

    # 2. Fetch comments for this question
    c_stmt = select(
        comments.c.id,
        comments.c.text,
        comments.c.created_at,
    ).where(comments.c.question_id == question_id).order_by(desc(comments.c.created_at))
    comment_result = await session.execute(c_stmt)
    comment_rows = comment_result.fetchall()

    comments_list = [
        CommentResponse(id=row.id, text=row.text, created_at=row.created_at)
        for row in comment_rows
    ]

    # 3. Fetch like/dislike counts for this question
    likes_dislikes_stmt = select(
        func.sum(sqlalchemy.case((question_likes.c.type == LikeDislikeType.LIKE.value, 1), else_=0)).label("likes_count"),
        func.sum(sqlalchemy.case((question_likes.c.type == LikeDislikeType.DISLIKE.value, 1), else_=0)).label("dislikes_count")
    ).where(question_likes.c.question_id == question_id)
    likes_dislikes_result = await session.execute(likes_dislikes_stmt)
    counts_row = likes_dislikes_result.first()

    likes_count = counts_row.likes_count if counts_row and counts_row.likes_count else 0
    dislikes_count = counts_row.dislikes_count if counts_row and counts_row.dislikes_count else 0

    return QuestionResponse(
        id=q_row.id,
        text=q_row.text,
        created_at=q_row.created_at,
        comments=comments_list,
        likes_count=likes_count,
        dislikes_count=dislikes_count
    )

async def create_comment_db(session: AsyncSession, question_id: str, text: str) -> CommentResponse:
    """
    Creates a new comment for a given question.
    Args:
        session (AsyncSession): The database session.
        question_id (str): The ID of the question to comment on.
        text (str): The text content of the comment.
    Returns:
        CommentResponse: The created comment's details.
    Raises:
        ValueError: If the newly created comment cannot be retrieved.
    """
    new_comment_id = str(uuid.uuid4()) # Generate UUID for comment ID
    query = comments.insert().values(
        id=new_comment_id,
        question_id=question_id,
        text=text,
        created_at=datetime.now()
    )
    await session.execute(query)
    # No commit here, handled by caller in route or service layer
    result = await session.execute(select(comments).where(comments.c.id == new_comment_id))
    created_comment = result.first()
    if created_comment:
        return CommentResponse(**created_comment._asdict())
    raise ValueError("Failed to retrieve the newly created comment.")

async def create_question_interaction_db(session: AsyncSession, question_id: str, user_id: int, interaction_type: LikeDislikeType) -> QuestionInteractionResponse:
    """
    Records a new like or dislike interaction for a question.
    Args:
        session (AsyncSession): The database session.
        question_id (str): The ID of the question being interacted with.
        user_id (int): The ID of the user performing the interaction.
        interaction_type (LikeDislikeType): The type of interaction ('like' or 'dislike').
    Returns:
        QuestionInteractionResponse: The created interaction's details.
    Raises:
        ValueError: If the newly created interaction cannot be retrieved.
    """
    new_interaction_id = str(uuid.uuid4())
    query = question_likes.insert().values(
        id=new_interaction_id,
        question_id=question_id,
        user_id=user_id,
        type=interaction_type.value, # Store the enum value (e.g., 'like' or 'dislike')
        created_at=datetime.now()
    )
    await session.execute(query)
    # Commit handled by caller
    result = await session.execute(select(question_likes).where(question_likes.c.id == new_interaction_id))
    created_interaction = result.first()
    if created_interaction:
        return QuestionInteractionResponse(**created_interaction._asdict())
    raise ValueError("Failed to retrieve the newly created interaction.")

async def get_user_question_interaction_db(session: AsyncSession, user_id: int, question_id: str) -> Optional[QuestionInteractionResponse]:
    """
    Retrieves a user's existing like/dislike interaction for a specific question.
    Args:
        session (AsyncSession): The database session.
        user_id (int): The ID of the user.
        question_id (str): The ID of the question.
    Returns:
        Optional[QuestionInteractionResponse]: The interaction details or None if not found.
    """
    stmt = select(question_likes).where(
        question_likes.c.user_id == user_id,
        question_likes.c.question_id == question_id
    )
    result = await session.execute(stmt)
    row = result.first()
    if row:
        return QuestionInteractionResponse(**row._asdict())
    return None

async def update_question_interaction_db(session: AsyncSession, interaction_id: str, new_type: LikeDislikeType) -> QuestionInteractionResponse:
    """
    Updates an existing like/dislike interaction.
    Args:
        session (AsyncSession): The database session.
        interaction_id (str): The ID of the interaction to update.
        new_type (LikeDislikeType): The new type of interaction.
    Returns:
        QuestionInteractionResponse: The updated interaction details.
    Raises:
        ValueError: If the updated interaction cannot be retrieved.
    """
    current_time = datetime.now()
    stmt = question_likes.update().where(question_likes.c.id == interaction_id).values(
        type=new_type.value, # Use .value for enum
        created_at=current_time # Update timestamp
    )
    await session.execute(stmt)
    # Commit handled by caller

    # Fetch the updated row to return
    updated_result = await session.execute(select(question_likes).where(question_likes.c.id == interaction_id))
    updated_row = updated_result.first()
    if updated_row:
        return QuestionInteractionResponse(**updated_row._asdict())
    raise ValueError("Updated interaction not found, something went wrong during update and fetch.")


async def delete_question_interaction_db(session: AsyncSession, interaction_id: str):
    """
    Deletes a like/dislike interaction.
    Args:
        session (AsyncSession): The database session.
        interaction_id (str): The ID of the interaction to delete.
    """
    stmt = question_likes.delete().where(question_likes.c.id == interaction_id)
    await session.execute(stmt)
    # Commit handled by caller


# --- User & Existence Checks (Helper Functions) ---

async def user_exists_db(session: AsyncSession, user_id: int) -> bool:
    """
    Checks if a user with the given ID exists.
    Args:
        session (AsyncSession): The database session.
        user_id (int): The ID of the user to check.
    Returns:
        bool: True if the user exists, False otherwise.
    """
    stmt = select(users.c.id).where(users.c.id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None

# --- Researcher CRUD Operations ---

async def create_researcher_db(session: AsyncSession, researcher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new researcher entry in the database.
    Args:
        session (AsyncSession): The database session.
        researcher_data (Dict[str, Any]): Dictionary containing researcher details.
    Returns:
        Dict[str, Any]: The created researcher's data.
    """
    new_id = str(uuid.uuid4())
    query = researchers.insert().values(
        id=new_id,
        title=researcher_data["title"],
        authors=researcher_data["authors"],
        publication_date=researcher_data["publication_date"],
        url=researcher_data["url"],
        profile=researcher_data["profile"], # Added to store the abstract/profile
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    await session.execute(query)
    await session.commit()
    result = await session.execute(select(researchers).where(researchers.c.id == new_id))
    return dict(result.first()) if result.first() else None

async def get_all_researchers_db(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Retrieves all researcher entries from the database.
    Args:
        session (AsyncSession): The database session.
    Returns:
        List[Dict[str, Any]]: A list of all researcher data.
    """
    # Explicitly select all expected columns, including 'profile'
    query = select(
        researchers.c.id,
        researchers.c.title,
        researchers.c.authors,
        researchers.c.publication_date,
        researchers.c.url,
        researchers.c.profile, # Ensure profile is selected
        researchers.c.created_at,
        researchers.c.updated_at
    ).order_by(desc(researchers.c.created_at))
    result = await session.execute(query)
    return [dict(row) for row in result.fetchall()]

async def get_researcher_by_id_db(session: AsyncSession, researcher_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single researcher entry by its ID.
    Args:
        session (AsyncSession): The database session.
        researcher_id (str): The ID of the researcher.
    Returns:
        Optional[Dict[str, Any]]: The researcher's data or None if not found.
    """
    # Explicitly select all expected columns, including 'profile'
    query = select(
        researchers.c.id,
        researchers.c.title,
        researchers.c.authors,
        researchers.c.publication_date,
        researchers.c.url,
        researchers.c.profile, # Ensure profile is selected
        researchers.c.created_at,
        researchers.c.updated_at
    ).where(researchers.c.id == researcher_id)
    result = await session.execute(query)
    return dict(result.first()) if result.first() else None

async def update_researcher_db(session: AsyncSession, researcher_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Updates an existing researcher entry in the database.
    Args:
        session (AsyncSession): The database session.
        researcher_id (str): The ID of the researcher to update.
        update_data (Dict[str, Any]): Dictionary of fields to update.
    Returns:
        Optional[Dict[str, Any]]: The updated researcher's data or None if update failed.
    """
    update_data["updated_at"] = datetime.now()
    query = researchers.update().where(researchers.c.id == researcher_id).values(**update_data)
    await session.execute(query)
    await session.commit()
    return await get_researcher_by_id_db(session, researcher_id)

async def delete_researcher_db(session: AsyncSession, researcher_id: str) -> bool:
    """
    Deletes a researcher entry from the database.
    Args:
        session (AsyncSession): The database session.
        researcher_id (str): The ID of the researcher to delete.
    Returns:
        bool: True if a row was deleted, False otherwise.
    """
    query = researchers.delete().where(researchers.c.id == researcher_id)
    result = await session.execute(query)
    await session.commit()
    return result.rowcount > 0 # Return True if a row was deleted
