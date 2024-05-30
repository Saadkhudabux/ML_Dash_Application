import pandas as pd
import dash
import pyodbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Connect to SQL Server
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=Analysis;Trusted_Connection=yes;')

# Define SQL query to fetch data
query = "SELECT * FROM dbo.master"

# Fetch data from SQL Server
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Suicide Analysis Dashboard", style={'textAlign': 'center', 'color': '#333333'}),
    
    html.Div([
        html.Label('Select Country:', style={'color': '#333333'}),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in df['country'].unique()],
            value='Japan',
            placeholder="Select a country",
            style={'width': '50%'},  # Adjusted width
        ),
    ], style={'textAlign': 'left', 'margin-bottom': '20px'}),
    
    html.Div([
        html.Label('Select Age Group:', style={'color': '#333333'}),
        dcc.Dropdown(
            id='age-dropdown',
            options=[{'label': 'All Age Groups', 'value': 'all'}] + [{'label': age_group, 'value': age_group} for age_group in df['age'].unique()],
            value='5-14 years', 
            placeholder="Select an age group",
            style={'width': '50%'},  # Adjusted width
        ),
    ], style={'textAlign': 'left', 'margin-bottom': '20px'}),
    
    html.Div([
        html.Label('Select Gender:', style={'color': '#333333'}),
        dcc.Dropdown(
            id='gender-dropdown',
            options=[
                {'label': 'All Genders', 'value': 'all'},
                {'label': 'Male', 'value': 'male'},
                {'label': 'Female', 'value': 'female'}
            ],
            value='all',
            placeholder="Select gender",
            style={'width': '50%'},  # Adjusted width
        ),
    ], style={'textAlign': 'left', 'margin-bottom': '20px'}),
    
    dcc.Graph(id='suicide-trend-graph'),
    
    dcc.Graph(id='suicide-age-group'),
    
    dcc.Graph(id='suicide-gender'),
    
    html.Div(id='prediction-output', style={'position': 'absolute', 'top': '20px', 'right': '20px', 'textAlign': 'right', })
])

# Define callback to update graphs and prediction based on dropdown selections
@app.callback(
    [Output('suicide-trend-graph', 'figure'),
     Output('suicide-age-group', 'figure'),
     Output('suicide-gender', 'figure'),
     Output('prediction-output', 'children')],
    [Input('country-dropdown', 'value'),
     Input('age-dropdown', 'value'),
     Input('gender-dropdown', 'value')]
)
def update_graph_and_prediction(selected_country, selected_age, selected_gender):
    # Apply filtering
    if selected_gender == 'all':
        if selected_age == 'all':
            filtered_df = df[(df['country'] == selected_country)]
        else:
            filtered_df = df[(df['country'] == selected_country) & (df['age'] == selected_age)]
    else:
        if selected_age == 'all':
            filtered_df = df[(df['country'] == selected_country) & (df['sex'] == selected_gender)]
        else:
            filtered_df = df[(df['country'] == selected_country) & (df['age'] == selected_age) & (df['sex'] == selected_gender)]
    
    # Convert 'year' column to integers
    filtered_df['year'] = filtered_df['year'].astype(int)
    
    fig1 = px.line(filtered_df, x='year', y='suicides_no', title=f'Suicides Over Time in {selected_country} ({selected_age}, {selected_gender})')
    fig1.update_traces(line=dict(width=2))
    
    if selected_age == 'all':
        fig2 = px.scatter(filtered_df, x='year', y='suicides_no', title=f'Suicides by Age Group in {selected_country} ({selected_gender})',
                    labels={'age': 'Age Group'}, category_orders={'age': ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']}, color_discrete_sequence=px.colors.sequential.Viridis)
        fig2.update_traces(marker=dict(opacity=0.7))
    else:
        fig2 = px.scatter(filtered_df, x='year', y='suicides_no', title=f'Suicides by Age Group in {selected_country} ({selected_age}, {selected_gender})',
                  labels={'age': 'Age Group'}, category_orders={'age': ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']}, color_discrete_sequence=px.colors.sequential.Viridis)
        fig2.update_traces(marker=dict(opacity=0.7))
    
    fig3 = px.pie(filtered_df, names='sex', values='suicides_no', title=f'Suicides by Gender in {selected_country} ({selected_age})',
                  hole=0.4)
    fig3.update_traces(textinfo='percent+label', pull=[0.1, 0])
    
    # Prepare data for prediction
    X = filtered_df[['year']]
    y = filtered_df['suicides_no']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future suicides
    future_year = max(filtered_df['year']) + 1
    future_data = pd.DataFrame({'year': [future_year]})
    future_suicides = model.predict(future_data)

    # Generate prediction text
    prediction_text = f'Projected Suicides in {future_year}: {int(future_suicides[0])}'

    return fig1, fig2, fig3, prediction_text

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
