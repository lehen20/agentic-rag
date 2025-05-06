import lancedb
import pyarrow as pa

def get_lancedb_table():
    db = lancedb.connect("my_lancedb")

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.struct([
            pa.field("producer", pa.string(), True),
            pa.field("creator", pa.string(), True),
            pa.field("creationdate", pa.string(), True),
            pa.field("source", pa.string(), True),
            pa.field("total_pages", pa.int32(), True),
            pa.field("page", pa.int32(), True),
            pa.field("page_label", pa.string(), True),
        ])),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ])
    print(f"Table: {db}")
    return db.create_table("docs", schema=schema, mode="overwrite"), db
