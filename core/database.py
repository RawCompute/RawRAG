import sqlite3

def initialize_database(path='rawrag.db'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            text TEXT
        )
    ''')
    connection.commit()
    connection.close()

def insert_chunk(chunks, start_id=0, path='rawrag.db'):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    data = [(start_id + i, text) for i, text in enumerate(chunks)]

    cursor.executemany('INSERT INTO chunks (id, text) VALUES (?, ?)', data)
    connection.commit()
    connection.close()

def get_chunks(id_list, db_path="rawrag.db"):
    if not id_list:
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the correct number of placeholders (?) for the SQL query
    placeholders = ','.join(['?'] * len(id_list))
    query = f'SELECT id, text FROM chunks WHERE id IN ({placeholders})'
    
    cursor.execute(query, id_list)
    results = cursor.fetchall()
    conn.close()
    
    # SQLite returns unordered rows. This dictionary maps them back to your ranked vector order.
    chunk_dict = {row[0]: row[1] for row in results}
    return [chunk_dict[i] for i in id_list if i in chunk_dict]
