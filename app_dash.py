import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import threading

# Load and preprocess dataset
data = pd.read_csv('chatbot_dataset.csv')
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Train the model
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment (optional)

# App layout
app.layout = html.Div([
    html.H1("💬 Chatbot", style={
        'textAlign': 'center', 'color': '#6482AD',
        'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '10px'
    }),

    dcc.Tabs([
        dcc.Tab(label='Chat', children=[
            html.Div([
                dcc.Textarea(
                    id='user-input',
                    placeholder='Type your question here...',
                    style={
                        'width': '100%', 'height': '100px', 'padding': '12px',
                        'borderRadius': '10px', 'border': '1px solid #ccc',
                        'fontSize': '16px', 'resize': 'none', 'fontFamily': 'Segoe UI'
                    }
                ),
                html.Button('Send', id='submit-button', n_clicks=0, style={
                    'backgroundColor': '#7FA1C3', 'color': 'white',
                    'border': 'none', 'padding': '12px 24px',
                    'marginTop': '10px', 'borderRadius': '8px',
                    'fontSize': '16px', 'cursor': 'pointer'
                }),
                html.Div(id='typing-indicator', style={
                    'marginTop': '10px', 'color': '#6482AD',
                    'fontStyle': 'italic', 'display': 'none'
                }),
                html.Div(id='chatbot-output', style={
                    'marginTop': '20px', 'backgroundColor': '#E2DAD6',
                    'padding': '20px', 'borderRadius': '12px',
                    'fontFamily': 'Segoe UI', 'fontSize': '16px'
                })
            ], style={
                'maxWidth': '600px', 'margin': '0 auto',
                'backgroundColor': '#F5EDED', 'padding': '20px',
                'borderRadius': '15px', 'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.1)'
            })
        ]),

        dcc.Tab(label='About', children=[
            html.Div([
                html.H3("🤖 About This Bot", style={'color': '#6482AD'}),
                html.P("This chatbot was built using a Naive Bayes classifier trained on a custom Q&A dataset. The UI is styled with Dash using pastel colors for a modern, calm experience.")
            ], style={
                'maxWidth': '600px', 'margin': '20px auto',
                'fontFamily': 'Segoe UI', 'fontSize': '16px'
            })
        ])
    ])
], style={'backgroundColor': '#F5EDED', 'minHeight': '100vh', 'padding': '20px'})


# Typing indicator handler
@app.callback(
    Output('typing-indicator', 'style'),
    Input('submit-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_typing(n_clicks):
    if n_clicks:
        return {'marginTop': '10px', 'color': '#6482AD', 'fontStyle': 'italic', 'display': 'block'}
    return {'display': 'none'}


# Main chatbot response logic
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, user_input):
    if user_input.strip() == "":
        return html.Div("Please enter a question.")
    
    # Simulate typing delay
    time.sleep(1)  # mimic processing delay
    question = ' '.join(nltk.word_tokenize(user_input.lower()))
    answer = model.predict([question])[0]

    return html.Div([
        html.Div(f"You: {user_input}", style={'marginBottom': '10px'}),
        html.Div(f"Bot: {answer}", style={
            'backgroundColor': '#ffffff', 'padding': '15px',
            'borderRadius': '10px', 'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.05)'
        })
    ])
    

# Run app
if __name__ == '__main__':
    app.run(debug=True)
