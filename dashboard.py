import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
import plotly.express as px
import pandas as pd
import base64
import io

# --- FIX 1: Add external_stylesheets to correctly load Font Awesome and prevent the UserWarning ---
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# --- App Layout ---
app.layout = html.Div([
    dcc.Store(id='stored-data'),
    html.Div([
        html.H1("Face Recognition Attendance System (SNJB)", className="header-title"),
        html.P("Upload your attendance file to visualize and analyze student attendance patterns", className="header-description")
    ], className="header"),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select an Attendance File')]),
            style={'width': '100%','height': '60px','lineHeight': '60px','borderWidth': '2px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '20px 0'},
            multiple=False
        ),
        html.Div(id='output-data-upload', style={'textAlign': 'center', 'padding': '10px'}),
    ], className="card"),
    html.Div(id='dashboard-container', style={'display': 'none'}, children=[
        html.Div([
            html.Div([html.Label("Date Range"),dcc.DatePickerRange(id='date-range')], className="card"),
            html.Div([html.Label("Select Branch"),dcc.Dropdown(id='class-selector', multi=True, placeholder="All Branches")], className="card"),
            html.Div([
                html.Label("Attendance Status"),
                dcc.Checklist(id='status-selector',options=[{'label': ' Present', 'value': 'Present'},{'label': ' Absent', 'value': 'Absent'},{'label': ' Late', 'value': 'Late'}],value=['Present', 'Absent', 'Late'],inline=True)
            ], className="card")
        ], className="filters"),
        html.Div([
            html.Div(id='total-students', className="metric-card"),
            html.Div(id='attendance-rate', className="metric-card"),
            html.Div(id='avg-daily', className="metric-card"),
            html.Div(id='absent-trend', className="metric-card")
        ], className="metrics"),
        html.Div([
            html.Div(dcc.Graph(id='attendance-trend-chart'), className="chart-card"),
            html.Div(dcc.Graph(id='daily-attendance-chart'), className="chart-card"),
            html.Div(dcc.Graph(id='class-comparison-chart'), className="chart-card"),
            html.Div([
                html.H3("Individual Student Records"),
                html.Div([
                    dcc.Input(id='student-search', placeholder='Search by student name...', type='text', debounce=True),
                    dash_table.DataTable(id='student-table',page_size=10,style_table={'overflowX': 'auto'},style_cell={'textAlign': 'left'},style_header={'fontWeight': 'bold'},)
                ])
            ], className="table-card")
        ], className="charts")
    ])
], className="container")

# --- Helper Function to Parse Uploaded File ---
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise Exception("Unsupported file type. Please upload a CSV or Excel file.")

        df.columns = df.columns.str.lower()
        required_cols = ['date', 'time', 'student_id', 'name', 'branch', 'status']
        if not all(col in df.columns for col in required_cols):
             raise Exception(f"File is missing one of the required columns: {required_cols}")

        column_mapping = {'date': 'Date','time': 'Time','student_id': 'Student ID','name': 'Student Name','branch': 'Class','status': 'Attendance Status'}
        df.rename(columns=column_mapping, inplace=True)
        
        # --- FIX 2: Make date conversion robust. This prevents crashes from bad date formats. ---
        # `errors='coerce'` will turn any invalid date strings into `NaT` (Not a Time)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop any rows where the date could not be parsed.
        df.dropna(subset=['Date'], inplace=True)
        
        df['Month'] = df['Date'].dt.month_name()
        df['Day'] = df['Date'].dt.day_name()
        df['Week'] = df['Date'].dt.isocalendar().week
        return df
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# --- Callbacks (No changes needed here) ---

@app.callback(
    Output('stored-data', 'data'),
    Output('output-data-upload', 'children'),
    Output('dashboard-container', 'style'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is None:
            error_message = html.Div(['There was an error processing this file. Please ensure it has the correct columns (date, time, student_id, name, branch, status) and is a valid CSV or Excel file.'], style={'color':'red'})
            return None, error_message, {'display': 'none'}
        return df.to_json(date_format='iso', orient='split'), html.Div(f'File "{filename}" processed successfully.'), {'display': 'block'}
    return None, "Please upload a file to begin.", {'display': 'none'}

@app.callback(
    Output('date-range', 'min_date_allowed'),
    Output('date-range', 'max_date_allowed'),
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date'),
    Output('class-selector', 'options'),
    Input('stored-data', 'data')
)
def update_filter_options(jsonified_dataframe):
    if jsonified_dataframe is None:
        return no_update
    df = pd.read_json(jsonified_dataframe, orient='split')
    df['Date'] = pd.to_datetime(df['Date'])
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    class_options = [{'label': cls, 'value': cls} for cls in sorted(df['Class'].unique())]
    return min_date, max_date, min_date, max_date, class_options

@app.callback(
    Output('attendance-trend-chart', 'figure'),
    Output('daily-attendance-chart', 'figure'),
    Output('class-comparison-chart', 'figure'),
    Output('student-table', 'data'),
    Output('total-students', 'children'),
    Output('attendance-rate', 'children'),
    Output('avg-daily', 'children'),
    Output('absent-trend', 'children'),
    Input('stored-data', 'data'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('class-selector', 'value'),
    Input('status-selector', 'value'),
    Input('student-search', 'value')
)
def update_dashboard(jsonified_dataframe, start_date, end_date, selected_classes, selected_statuses, search_query):
    if jsonified_dataframe is None:
        return {}, {}, {}, [], "", "", "", ""

    df = pd.read_json(jsonified_dataframe, orient='split')
    df['Date'] = pd.to_datetime(df['Date'])

    filtered_df = df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]
    if selected_classes:
        filtered_df = filtered_df[filtered_df['Class'].isin(selected_classes)]
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['Attendance Status'].isin(selected_statuses)]
    if search_query:
        filtered_df = filtered_df[filtered_df['Student Name'].str.contains(search_query, case=False, na=False)]
    
    if filtered_df.empty:
        empty_fig = {'layout': {'title': 'No data to display for the selected filters.'}}
        return empty_fig, empty_fig, empty_fig, [], "Total Students: 0", "Attendance Rate: N/A", "Avg Daily Attendance: N/A", "Absenteeism: N/A"

    trend_df = filtered_df.groupby(['Date', 'Attendance Status']).size().unstack(fill_value=0)
    trend_fig = px.line(trend_df, x=trend_df.index, y=trend_df.columns, title='Attendance Trends Over Time')
    trend_fig.update_layout(yaxis_title='Number of Students', legend_title_text='Status')
    
    daily_df = filtered_df.groupby(['Day', 'Attendance Status']).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_fig = px.bar(daily_df, category_orders={"Day": day_order}, title='Attendance by Day of Week', barmode='stack')
    daily_fig.update_layout(yaxis_title='Number of Students', legend_title_text='Status')
    
    class_df = filtered_df.groupby(['Class', 'Attendance Status']).size().unstack(fill_value=0).fillna(0)
    if 'Present' not in class_df.columns: class_df['Present'] = 0
    class_df['Total'] = class_df.sum(axis=1)
    class_df['Attendance Rate'] = (class_df['Present'] / class_df['Total'] * 100).round(1)
    class_fig = px.bar(class_df.reset_index().sort_values('Attendance Rate', ascending=False), x='Class', y='Attendance Rate',title='Attendance Rate by Branch',color='Attendance Rate', color_continuous_scale='Viridis', text='Attendance Rate')
    class_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    class_fig.update_layout(xaxis_title="Branch", yaxis_title='Attendance Rate (%)', yaxis_range=[0,110])

    table_df = filtered_df[['Date', 'Time', 'Student Name', 'Class', 'Attendance Status']].copy()
    table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')
    student_table = table_df.to_dict('records')
    
    total_students_val = filtered_df['Student ID'].nunique()
    total_students = [html.I(className="fas fa-users"), f" Total Students: {total_students_val}"]
    
    present_count = filtered_df[filtered_df['Attendance Status'] == 'Present'].shape[0]
    total_records = filtered_df.shape[0]
    att_rate_val = (present_count / total_records * 100) if total_records > 0 else 0
    attendance_rate = [html.I(className="fas fa-chart-pie"), f" Attendance Rate: {att_rate_val:.1f}%"]
    
    avg_daily_val = filtered_df.groupby('Date')['Student ID'].nunique().mean()
    avg_daily = [html.I(className="fas fa-calendar-day"), f" Avg Daily Presence: {avg_daily_val:.1f}"]
    
    absent_count = filtered_df[filtered_df['Attendance Status'] == 'Absent'].shape[0]
    absent_trend_icon = "fas fa-arrow-down" if absent_count < (total_records * 0.1) else "fas fa-arrow-up"
    absent_trend_text = " Improving" if absent_count < (total_records * 0.1) else " Needs Attention"
    absent_trend_color = 'var(--success)' if absent_count < (total_records * 0.1) else 'var(--danger)'
    absent_trend = [html.I(className=absent_trend_icon, style={'color': absent_trend_color}), " Absenteeism:", html.Span(absent_trend_text, style={'color': absent_trend_color})]

    return trend_fig, daily_fig, class_fig, student_table, total_students, attendance_rate, avg_daily, absent_trend

# --- CSS Styling (Same as before) ---
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Face Recognition Attendance System (SNJB)</title>
        {%favicon%}
        {%css%}
        <style>
            :root { --primary: #4e73df; --secondary: #858796; --success: #1cc88a; --warning: #f6c23e; --danger: #e74a3b; --light: #f8f9fc; --dark: #5a5c69; }
            body { font-family: 'Nunito', 'Segoe UI', sans-serif; background-color: var(--light); margin: 0; padding: 0; }
            .container { padding: 0 20px; }
            .header { background: linear-gradient(135deg, var(--primary), #2a4cb3); color: white; padding: 1.5rem 2rem; border-radius: 0 0 10px 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .header-title { margin: 0; font-weight: 700; }
            .header-description { opacity: 0.9; margin: 5px 0 0; }
            .filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .card { background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin: 20px 0; }
            .metric-card { background: white; padding: 20px; display: flex; align-items: center; justify-content: center; gap: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-weight: bold; font-size: 1.1rem; border-left: 5px solid var(--primary); }
            .metric-card .fas { color: var(--primary); }
            .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .chart-card, .table-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
            .table-card { grid-column: 1 / -1; }
            @media (max-width: 992px) { .charts { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body> {%app_entry%} <footer> {%config%} {%scripts%} {%renderer%} </footer> </body>
</html>'''

if __name__ == '__main__':
    app.run(debug=True, port='8051')