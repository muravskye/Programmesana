import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
DATABASE = 'database.db'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
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

def load_and_clean_csv(filepath):
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = None
            line_count = 0
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('Kopâ:'):
                    continue

                if header is None and 'Name' in line:
                    clean_h = line.replace('ē', 'e').replace('ā', 'a').replace('ī', 'i')\
                                  .replace('ļ', 'l').replace('ķ', 'k').replace('ģ', 'g')\
                                  .replace('š', 's').replace('ū', 'u')
                    clean_h = clean_h.replace('Energetiska vertiba', 'Calories')\
                                     .replace('Olbaltumvielas', 'Protein')\
                                     .replace('Tauki', 'Fat')\
                                     .replace('Oglhidrati', 'Carbs')\
                                     .replace('Sals', 'Salt')\
                                     .replace('Cukurs', 'Sugar')\
                                     .replace('Skiedrvielas', 'Fiber')
                    clean_h = clean_h.replace('(g)', '').replace('(kkal)', '')
                    header = [h.strip() for h in clean_h.split(', ')]
                    lines.append(header)
                    continue

                if header is not None:
                    cleaned_line = line
                    data_values = [val.strip() for val in cleaned_line.split(', ')]
                    if line_count < 5 or line_count % 20 == 0:
                        if len(data_values) != len(header):
                            print(f"!!! WARNING: Line {line_count} has {len(data_values)} values, expected {len(header)}")
                    lines.append(data_values)

    except FileNotFoundError:
        return None
    except Exception as e:
        return None

    if not lines or len(lines) < 2:
        return None

    try:
        header_list = lines[0]
        header_len = len(header_list)
        valid_data_lines = []
        invalid_line_count = 0
        for i, row in enumerate(lines[1:], start=2):
            if len(row) == header_len:
                valid_data_lines.append(row)
            else:
                invalid_line_count += 1

        if not valid_data_lines:
            return None

        df = pd.DataFrame(valid_data_lines, columns=header_list)

        expected_cols = ['Name', 'Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
        cols_to_keep = [col for col in expected_cols if col in df.columns]
        df = df[cols_to_keep]

        numeric_cols_to_convert = ['Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
        for col in numeric_cols_to_convert:
            if col in df.columns:
                original_dtype = df[col].dtype
                if pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        rows_to_drop = df[df['Name'].isnull() | df['Calories'].isnull()]
        if not rows_to_drop.empty:
            df = df.dropna(subset=['Name', 'Calories'])

        if df.empty:
            return None

        return df

    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = "data.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as e:
                return "Error saving uploaded file.", 500

            df = load_and_clean_csv(filepath)

            if df is not None and not df.empty:
                 init_db()
                 conn = get_db()
                 db_cols = ['Name', 'Weight', 'Calories', 'Protein', 'Fat', 'Carbs', 'Salt', 'Sugar', 'Fiber']
                 df_to_insert = df[[col for col in db_cols if col in df.columns]]
                 try:
                     df_to_insert.to_sql('meals', conn, if_exists='append', index=False)
                 except Exception as e:
                     pass
                 finally:
                    if conn:
                        conn.close()
                 return redirect(url_for('view_db_data'))
            else:
                 return "Error processing CSV file or file is empty/invalid. Check console logs for details.", 400
        else:
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/data')
def view_data_from_file():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
    plot_url = None
    table_html = "No data file found. Please upload a CSV file via the Upload page."
    df = None
    min_calories_filter = request.args.get('min_calories', default=0, type=float)

    if os.path.exists(filepath):
        df = load_and_clean_csv(filepath)

    if df is not None and not df.empty:
        df_filtered = df.copy()
        if min_calories_filter > 0:
             if 'Calories' in df_filtered.columns:
                 df_filtered['Calories'] = pd.to_numeric(df_filtered['Calories'], errors='coerce')
                 df_filtered = df_filtered[df_filtered['Calories'] >= min_calories_filter]
        if not df_filtered.empty and 'Calories' in df_filtered.columns:
            try:
                plt.figure(figsize=(8, 4))
                calories_data = df_filtered['Calories'].dropna()
                if not calories_data.empty:
                    plt.hist(calories_data, bins=10, color='skyblue', edgecolor='black')
                    plt.title('Histogram of Calories')
                    plt.xlabel('Calories (kcal)')
                    plt.ylabel('Frequency')
                    plt.tight_layout()

                    img = io.BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight')
                    plt.close()
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                else:
                    plot_url = None
            except Exception as e:
                 plot_url = None
                 plt.close()

        if not df_filtered.empty:
            table_html = df_filtered.to_html(classes='table table-striped', index=False, na_rep='-')
        else:
            table_html = "No data matching the filter."

    elif df is None and os.path.exists(filepath):
        table_html = "Error loading or processing the CSV file. Check console logs for details."
    elif df is not None and df.empty:
        table_html = "CSV file loaded, but contained no valid data rows after cleaning."

    return render_template('data.html', plot_url=plot_url, table=table_html, current_filter=min_calories_filter)

@app.route('/db_view')
def view_db_data():
    conn = None
    rows = []
    min_calories_filter = request.args.get('min_calories', default=0, type=float)

    try:
        conn = get_db()
        cursor = conn.cursor()

        query = "SELECT Name, Weight, Calories, Protein, Fat, Carbs, Salt, Sugar, Fiber FROM meals"
        params = []
        if min_calories_filter > 0:
            query += " WHERE Calories >= ?"
            params.append(min_calories_filter)

        query += " ORDER BY Protein DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

    except sqlite3.OperationalError as e:
        rows = []
    except Exception as e:
         rows = []
    finally:
        if conn:
            conn.close()

    data_list = [dict(row) for row in rows]
    df_from_db = pd.DataFrame(data_list)
    table_html = ""

    if not df_from_db.empty:
         table_html = df_from_db.to_html(classes='table table-striped', index=False, na_rep='-')
    elif not rows and min_calories_filter == 0:
         table_html = "Database is empty or the 'meals' table doesn't exist. Please upload a CSV file via the Upload page."
    else:
         table_html = "No data matching the filter found in the database."

    return render_template('db_view.html', table=table_html, current_filter=min_calories_filter)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
