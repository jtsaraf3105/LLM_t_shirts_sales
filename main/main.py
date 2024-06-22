import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
from langchain_helper import get_few_shot_db_chain

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment on some platforms

# CSS styles
styles = {
    'fontFamily': 'Overpass, sans-serif',
    'textAlign': 'center',
    'maxWidth': '800px',
    'margin': 'auto'
}

# Define app layout
app.layout = html.Div([
    html.H1("AtliQ T Shirts: Database Q&A 👕",
            style={'fontFamily': 'Overpass, sans-serif', 'fontSize': '2.5em', 'marginTop': '30px'}),
    dcc.Input(id='question-input', type='text', placeholder='Enter your question...',
              style={'width': '80%', 'height': '50px', 'margin': '20px auto', 'display': 'block',
                     'fontFamily': 'sans-serif', 'fontWeight': 'bold'}),
    html.Button('Submit', id='submit-button', n_clicks=0,
                style={'width': '80%', 'margin': '10px auto', 'display': 'block', 'background-color': '#4CAF50',
                       'color': 'white', 'padding': '14px 20px', 'border': 'none', 'cursor': 'pointer',
                       'borderRadius': '4px', 'fontFamily': 'Overpass, sans-serif',
                       'transition': 'background-color 0.3s, box-shadow 0.3s'}),
    html.Div(id='output-container', style={'margin': '20px auto', 'width': '80%', 'fontSize': '1.2em'}),
], style=styles)


# Define callback to update output based on input
@app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('question-input', 'value')]
)
def update_output(n_clicks, question):
    if n_clicks == 0:
        raise PreventUpdate

    if question:
        try:
            chain = get_few_shot_db_chain()
            response = chain.run(question)
            return html.Div([
                html.H3("Answer", style={'color': '#333333', 'textAlign': 'center', 'fontSize': '1.5em'}),
                html.P(response, style={'color': '#666666', 'textAlign': 'center'})
            ])
        except Exception as e:
            return html.Div([
                html.P("Error processing the question. Please try again.",
                       style={'color': 'red', 'textAlign': 'center'})
            ])
    else:
        return html.Div()


if __name__ == '__main__':
    app.run_server(debug=True)
