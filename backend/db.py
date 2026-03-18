from databricks.sdk import WorkspaceClient
import uuid

# Initialize Databricks workspace client (native authentication)
w = WorkspaceClient()

# Replace with your SQL Warehouse ID
WAREHOUSE_ID = "afe5734e6076a678"


# -------------------------------
# Fetch all cases (Case List Page)
# -------------------------------
def get_all_cases():
    query = """
    SELECT case_id,
           title,
           status,
           created_date
    FROM msci.ticket_management_system.support_cases
    ORDER BY created_date DESC
    """

    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID
    )

    return response.result.data_array


# -------------------------------
# Fetch single case (Case Detail Page)
# -------------------------------
def get_case_by_id(case_id):
    query = f"""
    SELECT *
    FROM msci.ticket_management_system.support_cases
    WHERE case_id = '{case_id}'
    """

    response = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID
    )

    data = response.result.data_array
    return data[0] if data else None


# -------------------------------
# Create new case
# -------------------------------
def create_case(title, created_by):
    case_id = str(uuid.uuid4())

    query = f"""
    INSERT INTO msci.ticket_management_system.support_cases
    (case_id, title, status, created_by, created_date)
    VALUES
    ('{case_id}', '{title}', 'Open', '{created_by}', current_timestamp())
    """

    w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID
    )

    return case_id


# -------------------------------
# Update existing case
# -------------------------------
def update_case(case_id, title, description, status, assigned_to, modified_by):
    query = f"""
    UPDATE msci.ticket_management_system.support_cases
    SET title = '{title}',
        description = '{description}',
        status = '{status}',
        assigned_to = '{assigned_to}',
        modified_by = '{modified_by}',
        modified_date = current_timestamp()
    WHERE case_id = '{case_id}'
    """

    w.statement_execution.execute_statement(
        statement=query,
        warehouse_id=WAREHOUSE_ID
    )

    return True