# Here are several methods to inspect data stored in LanceDB

import lancedb
import pandas as pd

# 1. Connect to your LanceDB database
db_uri = "my_lancedb"  # Or the path you're using
connection = lancedb.connect(db_uri)

# 2. List all tables in the database
tables = connection.table_names()
print(f"Tables in database: {tables}")

# 3. Open a specific table
table_name = "docs"  # Or whatever name you used
if table_name in tables:
    table = connection.open_table(table_name)

    # 4. Get table schema
    schema = table.schema
    print(f"Table schema: {schema}")

    # 5. Count total records
    count = len(table)
    print(f"Total records: {count}")

    # 6. View the first few records (without vector data for clarity)
    # Option A: To pandas DataFrame
    df = table.to_pandas()
    # # Remove vector column for cleaner display
    if "vector" in df.columns:
        df_display = df.drop(columns=["vector"])
    else:
        df_display = df
    print("\nFirst 5 records:")
    print(df_display.head())

    # 7. Query specific fields
    text_only = table.select(["text", "source"]).to_pandas()
    print("\nJust text and source fields:")
    print(text_only)

    # 8. Filter data with a condition
    filtered = table.filter("page == 0").to_pandas()
    if "vector" in filtered.columns:
        filtered = filtered.drop(columns=["vector"])
    print("\nFiltered data (page == 0):")
    print(filtered)

    # 9. Check for specific values
    if "source" in df.columns:
        sources = df["source"].unique()
        print(f"\nUnique source values: {sources}")

    # 10. Get metadata statistics if you have numeric fields
    if "page" in df.columns:
        page_stats = {
            "min": df["page"].min(),
            "max": df["page"].max(),
            "avg": df["page"].mean(),
            "count": df["page"].count(),
        }
        print(f"\nPage statistics: {page_stats}")
else:
    print(f"Table '{table_name}' not found")
