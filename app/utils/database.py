import sqlite3


class PredictionDB:
    def __init__(self, db_file="predictions.db"):
        """Initialize the class with the database file."""
        self.db_file = db_file
        self.create_table()  # Ensure the table is created when the object is initialized

    def create_connection(self):
        """Create a connection to the SQLite database."""
        return sqlite3.connect(self.db_file)

    def create_table(self):
        """Create the predictions table if it doesn't exist."""
        conn = self.create_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_planet TEXT,
                cryo_sleep TEXT,
                destination TEXT,
                age INTEGER,
                vip TEXT,
                total_spend REAL,
                transported TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def insert_prediction(
        self, home_planet, cryo_sleep, destination, age, vip, total_spend, transported
    ):
        """Insert a prediction into the database."""
        conn = self.create_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (home_planet, cryo_sleep, destination, age, vip, total_spend, transported)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (home_planet, cryo_sleep, destination, age, vip, total_spend, transported),
        )
        conn.commit()
        conn.close()

    def fetch_predictions(self):
        """Fetch all predictions from the database."""
        conn = self.create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        predictions = cursor.fetchall()
        conn.close()
        return predictions
