# import sqlite3

# DB_PATH = "cameras.db"

# def add_cameras():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     # Create table if not exists
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS Camera (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             name TEXT NOT NULL,
#             ip TEXT NOT NULL
#         )
#     """)
#     # Insert cameras from 11 to 26
#     for i in range(11, 27):
#         name = f"Camera_{i}"
#         ip = f"192.168.3.{i}"
#         c.execute("INSERT INTO Camera (name, ip) VALUES (?, ?)", (name, ip))
#     conn.commit()
#     conn.close()
#     print("Cameras added.")

# if __name__ == "__main__":

#     add_cameras()



import sqlite3

DB_PATH = "cameras.db"

def add_cameras():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS Camera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ip TEXT NOT NULL
        )
    """)
    # Insert only the specified cameras
    camera_ips = ["192.168.3.26", "192.168.3.16", "192.168.3.14"]
    for idx, ip in enumerate(camera_ips, start=1):
        name = f"Camera_{idx}"
        c.execute("INSERT INTO Camera (name, ip) VALUES (?, ?)", (name, ip))
    conn.commit()
    conn.close()
    print("Specified cameras added.")

if __name__ == "__main__":
    add_cameras()