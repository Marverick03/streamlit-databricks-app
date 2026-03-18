from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def get_all_cases():
    query = """
    SELECT * FROM support_app.cases
    ORDER BY created_date DESC
    """
    
    result = w.statement_execution.execute_statement(
        statement=query,
        warehouse_id="afe5734e6076a678"
    )
    
    return result
