def construct_prompt_lora(sql, duration, log):
    sql_prompt_lora = f'''
Task: Analyze the root cause of the following slow SQL query using the provided execution log and duration.
Rank potential root causes by their estimated impact on query performance.

Execution Log:
{log}

SQL Execution Time: {duration}

Slow SQL Query:
{sql}

Root Cause Categories (rank these in order of impact):
1. Outdated statistical information
2. Under-optimized join order
3. Inappropriate distribution keys
4. Missing indexes
5. Redundant indexes
6. Repeatedly executing subqueries
7. Complex table joins
8. Updating an entire table
9. Inserting large data

Analysis Format:
1. Root Cause: [Category Number] - [Brief Description]
   Evidence: [Specific details from the log or query]
   Impact: [High/Medium/Low - Justify based on execution time/dependencies]

2. Root Cause: [...]
   Evidence: [...]
   Impact: [...]

<CLS>'''
    return sql_prompt_lora
