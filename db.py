from databricks import sql
import os
import uuid

def get_connection():
    return sql.connect(
        server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path = os.getenv("DATABRICKS_HTTP_PATH"),
        access_token = os.getenv("DATABRICKS_TOKEN")
    )

def create_case(title, created_by):
    conn = get_connection()
    cursor = conn.cursor()

    case_id = str(uuid.uuid4())

    query = f"""
    INSERT INTO support_app.cases
    (case_id, title, status, created_by, created_date)
    VALUES
    ('{case_id}', '{title}', 'Open', '{created_by}', current_timestamp())
    """

    cursor.execute(query)
    conn.close()
    return case_id

def get_all_cases():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM support_app.cases ORDER BY created_date DESC")
    rows = cursor.fetchall()
    conn.close()

    return rows