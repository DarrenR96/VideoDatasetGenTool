import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import os
from dash.dependencies import Output,Input

app = dash.Dash(__name__)

def getfileName(x):
    _, x = os.path.split(x)
    return x

transcodesDF = pd.read_csv("DataOutput/suggestedEncodes.csv")
transcodesDF['fileName'] = transcodesDF['VideoName'].apply(lambda x: getfileName(x))
fileNames = [{'label':x, 'value':x} for x in transcodesDF['fileName'].unique()]


app.layout = html.Div(children=[
    html.H1(children='Suggested Transcodes'),
    dcc.Dropdown(
        id='file-dropdown',
        options=fileNames,
        value=fileNames[0]['value']
    ),
    html.Div([
        dcc.Slider(id='iterationSlider', min=0, step=1, value=0),
        html.Div(id='sliderNum')
    ]),
    dcc.Graph(
        id='scatter-plot'
    )
])

@app.callback(
    [
        Output('scatter-plot', 'figure'),
        Output('iterationSlider','max'),
    ],
    [
        Input('file-dropdown', 'value')
    ]
)
def updateScatter(fileName):
    selectedDF = transcodesDF[transcodesDF['fileName'] == fileName]
    maxIters = selectedDF['Iteration'].max()
    selectedDFCriteria = selectedDF[selectedDF['CriteriaExists'] == 1]
    selectedCRF = selectedDFCriteria['CRF'].tolist() 
    selectedVMAF = selectedDFCriteria['VMAF'].tolist() 
    upper = selectedDF['UpperTarget'].unique().tolist()
    lower = selectedDF['LowerTarget'].unique().tolist()
    fig = px.scatter(
        selectedDF, x='CRF', y='VMAF', color='Iteration', title= f"Evaluating criteria for VMAF between {lower[0]} <-> {upper[0]}", color_continuous_scale='agsunset'
    )
    fig.update_xaxes(range=[-5,55])
    fig.update_yaxes(range=[-5,105])
    if selectedVMAF:
        for (x,y) in zip(selectedCRF, selectedVMAF):
            fig.add_annotation(x=x, y=y, text="Solution Found", showarrow=True, arrowhead=1)
    return fig, maxIters

@app.callback(
    [
        Output('sliderNum','children')
    ],
    [
        Input('iterationSlider', 'value')
    ]
)
def updateSliderValue(x):
    x = f"Selected Iteration : {x}"
    return [x]

if __name__ == '__main__':
    app.run_server(debug=True)