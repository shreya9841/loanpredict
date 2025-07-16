import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

c.execute("SELECT * FROM users")
users = c.fetchall()

print("Stored Users:")
for user in users:
    print(user)

conn.close()
