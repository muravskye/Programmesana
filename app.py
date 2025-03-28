import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for

# Make sure matplotlib uses a backend that doesn't require a GUI
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, template_folder='templates') # Explicitly set template folder
UPLOAD_FOLDER = 'uploads'
DATABASE = 'database.db'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    print(f"Creating uploads directory at: {os.path.abspath(UPLOAD_FOLDER)}")
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    """Initializes the database by dropping and recreating the meals table."""
    conn = get_db()
    cursor = conn.cursor()
    print("Initializing DB: Dropping and creating 'meals' table.")
    cursor.execute("DROP TABLE IF EXISTS meals")
    cursor.execute("""
        CREATE TABLE meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Weight REAL,
            Calories REAL,
            Protein REAL,
            Fat REAL,
            Carbs REAL,
            Salt REAL,
            Sugar REAL,
            Fiber REAL
        )
    """)
    conn.commit()
    conn.close()
    print("DB Initialized.")

# --- CORRECTED load_and_clean_csv FUNCTION ---
def load_and_clean_csv(filepath):
    """Loads, cleans, and parses the specific CSV format into a Pandas DataFrame."""
    print(f"\n--- Starting load_and_clean_csv for: {filepath} ---")
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = None
            line_count = 0
            print("Reading file line by line...")
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('Kopâ:'):
                    # print(f"Skipping Line {line_count}: Empty or starts with 'Kopâ:'") # Optional: verbose
                    continue

                if header is None and 'Name' in line:
                    print(f"Found potential header on Line {line_count}: {line}")
                    # --- CORRECTED HEADER CLEANING ---
                    # 1. Replace Accents
                    clean_h = line.replace('ē', 'e').replace('ā', 'a').replace('ī', 'i')\
                                  .replace('ļ', 'l').replace('ķ', 'k').replace('ģ', 'g')\
                                  .replace('š', 's').replace('ū', 'u')
                    # 2. Replace Full Latvian Terms FIRST
                    clean_h = clean_h.replace('Energetiska vertiba', 'Calories')\
                                     .replace('Olbaltumvielas', 'Protein')\
                                     .replace('Tauki', 'Fat')\
                                     .replace('Oglhidrati', 'Carbs')\
                                     .replace('Sals', 'Salt')\
                                     .replace('Cukurs', 'Sugar')\
                                     .replace('Skiedrvielas', 'Fiber') # Target accent-cleaned version
                    # 3. Remove Units
                    clean_h = clean_h.replace('(g)', '').replace('(kkal)', '')
                    # 4. Split
                    header = [h.strip() for h in clean_h.split(', ')]
                    # --- END OF CORRECTIONS ---

                    print(f"Cleaned Header: {header}")
                    # Basic validation: Check if essential columns are roughly there
                    if not all(col in header for col in ['Name', 'Weight', 'Calories']):
                         print(f"!!! WARNING: Header cleaning might be incomplete: {header}")

                    lines.append(header)
                    continue

                if header is not None:
                    # --- REMOVED FAULTY LINE ---
                    # cleaned_line = line.replace(',', '.') # THIS WAS THE PROBLEM FOR DATA LINES
                    cleaned_line = line # Use the line directly as it's already correctly formatted

                    data_values = [val.strip() for val in cleaned_line.split(', ')]
                    # Print only the first few data lines splits for brevity
                    if line_count < 5 or line_count % 20 == 0 : # Print first few and every 20th
                         print(f"Split data values (Line {line_count}): {data_values}")
                         if len(data_values) != len(header):
                             print(f"!!! WARNING: Line {line_count} has {len(data_values)} values, expected {len(header)}")
                    lines.append(data_values)

    except FileNotFoundError:
        print(f"!!! ERROR: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"!!! ERROR reading file {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not lines or len(lines) < 2:
        print("!!! ERROR: No valid header or data lines found after reading file.")
        return None # No valid data found

    print(f"\nFinished reading file. Found {len(lines)-1} data lines.")

    # Create DataFrame, handling potential mismatch in column numbers if any line was bad
    try:
        print("\nAttempting to create DataFrame...")
        header_list = lines[0] # Get the processed header
        header_len = len(header_list)
        valid_data_lines = []
        invalid_line_count = 0
        for i, row in enumerate(lines[1:], start=2): # start=2 for correct line number reporting
            if len(row) == header_len:
                valid_data_lines.append(row)
            else:
                invalid_line_count += 1
                if invalid_line_count <= 5: # Print first few errors
                     print(f"!!! WARNING: Dropping row (original line approx {i}) due to incorrect column count ({len(row)} instead of {header_len}). Row data: {row}")
                elif invalid_line_count == 6:
                     print("!!! WARNING: More rows dropped due to incorrect column count (further warnings suppressed).")


        if not valid_data_lines:
            print("!!! ERROR: No data lines remain after filtering for correct column count.")
            return None

        df = pd.DataFrame(valid_data_lines, columns=header_list) # Use the processed header
        print("DataFrame created successfully. Initial rows:")
        print(df.head())
        print("\nDataFrame Info (Initial):")
        # Use StringIO to capture df.info() output as it prints to stdout
        buffer = io.StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())


        # Select only the columns defined in the DB schema
        expected_cols = ['Name', 'Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
        print(f"\nExpected DB columns: {expected_cols}")
        # Filter df columns based on expected_cols, handling potential missing columns gracefully
        cols_to_keep = [col for col in expected_cols if col in df.columns]
        if len(cols_to_keep) < len(expected_cols):
            missing_cols = [col for col in expected_cols if col not in df.columns]
            print(f"!!! WARNING: The following expected columns were NOT found in the CSV header after cleaning: {missing_cols}")
        df = df[cols_to_keep]
        print(f"DataFrame columns after selecting expected: {df.columns.tolist()}")


        # Convert numeric columns, coercing errors to NaN
        print("\nConverting numeric columns (errors='coerce')...")
        numeric_cols_to_convert = ['Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
        for col in numeric_cols_to_convert:
             if col in df.columns:
                # print(f"Converting column: {col}") # Less verbose
                original_dtype = df[col].dtype
                # Attempt conversion, first replacing empty strings with NaN if necessary
                # Handle potential non-string values before replace
                if pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check if NaNs were introduced ONLY during this conversion step
                if original_dtype == 'object':
                     # Create a boolean mask of original non-numeric-like strings (excluding empty/NA)
                     original_non_numeric_mask = ~pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce').notna() & df[col].notna()
                     # Count where it was originally non-numeric but is now NaN
                     nan_introduced_count = (original_non_numeric_mask & df[col].isnull()).sum()
                     if nan_introduced_count > 0:
                           print(f"  -> WARNING: ~{nan_introduced_count} NaN(s) introduced in column '{col}' from non-numeric values during conversion.")

        print("\nDataFrame Info (After Numeric Conversion):")
        buffer = io.StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())


        print(f"\nShape before dropping NaNs in 'Name' or 'Calories': {df.shape}")
        # Find rows that *will be* dropped
        rows_to_drop = df[df['Name'].isnull() | df['Calories'].isnull()]
        if not rows_to_drop.empty:
             print(f"!!! WARNING: Dropping {len(rows_to_drop)} rows due to NaN in 'Name' or 'Calories'. Example rows:")
             # Only print head if many rows are dropped
             print(rows_to_drop.head())


        df = df.dropna(subset=['Name', 'Calories']) # Remove rows where Name or Calories are missing after conversion
        print(f"Shape after dropping NaNs: {df.shape}")


        if df.empty:
            print("!!! ERROR: DataFrame is empty after dropping NaN rows.")
            return None

        print("\n--- load_and_clean_csv finished successfully ---")
        return df

    except Exception as e:
         # Basic error handling for DataFrame creation failure
         print(f"!!! EXCEPTION during DataFrame creation/processing: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for detailed debugging
         return None
# --- END OF CORRECTED FUNCTION ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handles file upload (POST) and renders the upload form (GET)."""
    print(f"\n--- Request to /upload (Method: {request.method}) ---")
    if request.method == 'POST':
        print("Handling POST request...")
        # Check if the post request has the file part
        if 'file' not in request.files:
            print("No 'file' part in request.files")
            # Add flash message?
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print("No file selected (filename is empty)")
            # Add flash message?
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = "data.csv" # Always save as data.csv, overwriting previous
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Attempting to save file '{file.filename}' to '{filepath}'")
            try:
                file.save(filepath)
                print("File saved successfully.")
            except Exception as e:
                print(f"!!! ERROR saving file: {e}")
                return "Error saving uploaded file.", 500


            # Load data and insert into DB
            print("Calling load_and_clean_csv...")
            df = load_and_clean_csv(filepath) # Use the corrected function

            if df is not None and not df.empty:
                 print("CSV loaded successfully into DataFrame. Proceeding to DB insertion.")
                 init_db() # Clear existing table before inserting new data
                 conn = get_db()
                 # Ensure only columns that exist in the DB table schema are inserted
                 db_cols = ['Name', 'Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
                 df_to_insert = df[[col for col in db_cols if col in df.columns]]
                 print(f"Columns to insert into DB: {df_to_insert.columns.tolist()}")
                 print(f"Number of rows to insert: {len(df_to_insert)}")
                 try:
                     print("Attempting df.to_sql...")
                     df_to_insert.to_sql('meals', conn, if_exists='append', index=False)
                     print("Data inserted into 'meals' table successfully.")
                 except Exception as e:
                     print(f"!!! ERROR inserting data into DB: {e}")
                     # Consider rendering an error page instead of just printing
                     # Or add a flash message for the user
                 finally:
                    if conn:
                        conn.close()
                        print("DB connection closed after insertion.")
                 print("Redirecting to view_db_data.")
                 return redirect(url_for('view_db_data')) # Redirect after successful upload
            else:
                 print("!!! load_and_clean_csv returned None or empty DataFrame.")
                 # Handle case where CSV loading failed or resulted in empty dataframe
                 # Add flash message?
                 return "Error processing CSV file or file is empty/invalid. Check console logs for details.", 400
        else:
            print(f"File '{file.filename}' not allowed (invalid extension).")
            # Add flash message?
            return redirect(request.url)

    # Render the upload form for GET requests
    print("Rendering upload.html template for GET request.")
    return render_template('upload.html')

@app.route('/data')
def view_data_from_file():
    """Displays data loaded directly from the last uploaded CSV file and a histogram."""
    print("\n--- Request to /data ---")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
    plot_url = None
    table_html = "No data file found. Please upload a CSV file via the Upload page."
    df = None
    # Initialize min_calories_filter early with a default
    min_calories_filter = request.args.get('min_calories', default=0, type=float)
    print(f"Min calories filter from request args: {min_calories_filter}")


    if os.path.exists(filepath):
        print(f"Data file exists: {filepath}. Loading using load_and_clean_csv...")
        df = load_and_clean_csv(filepath) # Reuse the cleaning logic for consistency
    else:
        print(f"Data file not found: {filepath}")


    if df is not None and not df.empty:
        print(f"DataFrame loaded successfully from file. Shape: {df.shape}")
        # --- Filtering ---
        # min_calories_filter is already defined
        print(f"Applying filter: Calories >= {min_calories_filter}")
        df_filtered = df.copy() # Start with a copy
        if min_calories_filter > 0:
             if 'Calories' in df_filtered.columns:
                 # Ensure Calories is numeric before filtering
                 df_filtered['Calories'] = pd.to_numeric(df_filtered['Calories'], errors='coerce')
                 df_filtered = df_filtered[df_filtered['Calories'] >= min_calories_filter]
             else:
                  print("!!! WARNING: 'Calories' column not found for filtering.")
        print(f"DataFrame shape after filtering: {df_filtered.shape}")


        # --- Histogram ---
        if not df_filtered.empty and 'Calories' in df_filtered.columns:
            print("Generating histogram...")
            try:
                plt.figure(figsize=(8, 4))
                # Ensure Calories column is numeric before plotting and drop NAs introduced by coerce
                calories_data = df_filtered['Calories'].dropna() # Already numeric from filtering step
                if not calories_data.empty:
                    plt.hist(calories_data, bins=10, color='skyblue', edgecolor='black')
                    plt.title('Histogram of Calories')
                    plt.xlabel('Calories (kcal)')
                    plt.ylabel('Frequency')
                    plt.tight_layout() # Adjust plot to prevent labels overlapping

                    img = io.BytesIO() # Create a BytesIO object
                    plt.savefig(img, format='png', bbox_inches='tight') # Save plot to the object
                    plt.close() # Close the figure to free memory
                    img.seek(0) # Go to the beginning of the BytesIO stream
                    plot_url = base64.b64encode(img.getvalue()).decode('utf8') # Encode image to base64
                    print("Histogram generated successfully.")
                else:
                    print("No valid numeric data in 'Calories' column for histogram after filtering.")

            except Exception as e:
                 print(f"!!! ERROR generating histogram: {e}")
                 plot_url = None # Ensure plot_url is None if error occurs
                 plt.close() # Attempt to close figure even if error occurred

        else:
             print("Filtered DataFrame is empty or 'Calories' column missing, skipping histogram.")


        # --- Data Table ---
        if not df_filtered.empty:
            print("Generating HTML table for filtered data...")
            table_html = df_filtered.to_html(classes='table table-striped', index=False, na_rep='-') # Display NaN as '-'
        else:
            print("Filtered data is empty, setting table message.")
            table_html = "No data matching the filter."

    elif df is None and os.path.exists(filepath):
        # Only show this error if the file existed but parsing failed
        print("load_and_clean_csv returned None.")
        table_html = "Error loading or processing the CSV file. Check console logs for details."
    elif df is not None and df.empty: # df is not None but is empty
        print("load_and_clean_csv returned an empty DataFrame initially.")
        table_html = "CSV file loaded, but contained no valid data rows after cleaning."
    # else: file doesn't exist, initial message remains


    print("Rendering data.html template.")
    # Ensure current_filter is passed correctly to the template
    return render_template('data.html', plot_url=plot_url, table=table_html, current_filter=min_calories_filter)


@app.route('/db_view')
def view_db_data():
    """Displays data queried directly from the SQLite database."""
    print("\n--- Request to /db_view ---")
    conn = None # Initialize conn
    rows = [] # Initialize rows
    # Initialize min_calories_filter early with a default
    min_calories_filter = request.args.get('min_calories', default=0, type=float)
    print(f"Min calories filter from request args: {min_calories_filter}")

    try:
        conn = get_db()
        cursor = conn.cursor()

        # --- Filtering (Query 1: Basic Filter) ---
        # Select all relevant columns from the database table
        query = "SELECT Name, Weight, Calories, Protein, Fat, Carbs, Salt, Sugar, Fiber FROM meals"
        params = []
        if min_calories_filter > 0:
            query += " WHERE Calories >= ?" # Add filter condition if needed
            params.append(min_calories_filter)

        # --- Query 2: Order by Protein (Example of a second query type) ---
        query += " ORDER BY Protein DESC" # Add ordering

        print(f"Executing DB query: {query} with params: {params}")
        cursor.execute(query, params)
        rows = cursor.fetchall() # Fetch all matching rows
        print(f"Query executed. Fetched {len(rows)} rows.")

    except sqlite3.OperationalError as e:
        # Handle case where table might not exist yet (e.g., first run)
        print(f"!!! DB OperationalError: {e}. Likely table 'meals' doesn't exist yet.")
        rows = []
    except Exception as e:
         print(f"!!! ERROR during DB query: {e}")
         rows = [] # Ensure rows is empty on other errors too
    finally:
        if conn:
            conn.close() # Ensure connection is closed
            print("DB connection closed.")


    # Convert fetched rows (sqlite3.Row objects) to a list of dictionaries
    data_list = [dict(row) for row in rows]
    # Create a DataFrame from the list of dictionaries
    df_from_db = pd.DataFrame(data_list)
    table_html = "" # Initialize table_html variable

    # Generate HTML table from the DataFrame
    if not df_from_db.empty:
         print("Generating HTML table from DB data...")
         table_html = df_from_db.to_html(classes='table table-striped', index=False, na_rep='-') # Display NaN as '-'
    elif not rows and min_calories_filter == 0: # No rows found and no filter applied
         print("DB is empty (or table missing) and no filter applied.")
         table_html = "Database is empty or the 'meals' table doesn't exist. Please upload a CSV file via the Upload page."
    else: # Either rows is empty with a filter, or df_from_db creation failed (unlikely here)
         print("No data matching the filter found in the database.")
         table_html = "No data matching the filter found in the database."

    print("Rendering db_view.html template.")
    # Pass the HTML table and current filter value to the template
    return render_template('db_view.html', table=table_html, current_filter=min_calories_filter)

if __name__ == '__main__':
    print("Starting Flask application...")
    init_db() # Initialize DB (creates table if not exists) when app starts
    print(f"Flask app '{app.name}' running with debug mode.")
    # template_dir = os.path.abspath('./templates') # Optional: confirm template path
    # print(f"Looking for templates in: {template_dir}")
    app.run(debug=True) # debug=True enables auto-reloading and debugger