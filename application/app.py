import warnings
warnings.filterwarnings('ignore')

import dash
import dash_daq as daq

import os
import numpy as np
import time
import cv2 as cv

import plotly.express as px
import plotly.graph_objects as go

from dash import html, dcc
from dash.dependencies import (
    Input, Output
)

########################################
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.title = 'DeepGlobe Road Extraction'
server = app.server
########################################

app.layout = html.Div([

    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),

    html.H3('DeepGlobe Road Extraction', style={'textAlign' : 'center'}),

    html.Div([
        html.Div([
            html.Div([
                html.P('Satellite Image'),
                dcc.Dropdown(
                    id='image-dropdown',
                    options=[
                        {'label' : '100034_sat', 'value' : '100034_sat.jpg'},
                        {'label' : '100393_sat', 'value' : '100393_sat.jpg'}
                    ],
                    value='100034_sat.jpg'
                )
            ]),

            html.Div([
                html.P('U-NET Model'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label' : 'No Augmentation', 'value' : 'unet_scratch'},
                        {'label' : 'Augmentation', 'value' : 'unet_scratch_augmentated'}
                    ],
                    value='unet_scratch'
                )
            ], style={'paddingTop' : 30}),

            html.Div([
                daq.ToggleSwitch(
                    id='extraction-mode',
                    size=60,
                    label='Extract Path',
                    labelPosition='top',
                    color='red',
                    value=False,
                )
            ], style={'paddingTop' : 50})
        ], className='five columns'),

        html.Div([
            html.Div([
                html.Div(id='image-prediction')
            ])
        ], className='seven columns')

    ], className='row', style={'paddingTop' : 30})
], className='container')


@app.callback(
    Output('image-prediction', 'children'),
    [Input('image-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('extraction-mode', 'value')
    ]
)
def extrach_road_path(sat_image_name, model_type, is_on):
    # I have already trained the U-NET model that can extract the path
    # Please check the notebook (04_CS2_Final_Submission.ipynb) files for more details
    
    app_path = os.getcwd().split('\\')

    if not is_on:
        sat_image_path = '\\'.join(app_path[:-1]) + '\\test_images\\{}'.format(sat_image_name)

        sat_image = cv.imread(sat_image_path)
        sat_image = cv.cvtColor(sat_image, cv.COLOR_BGR2RGB)

        sat_image_fig = px.imshow(sat_image)
        sat_image_fig.update_layout(
            coloraxis_showscale=False,
            autosize=True, height=400,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        sat_image_fig.update_xaxes(showticklabels=False)
        sat_image_fig.update_yaxes(showticklabels=False)

        output_result = html.Div([
            dcc.Graph(id='sat-image', figure=sat_image_fig)
        ])

        return output_result

    time.sleep(3)
    
    pred_images_path = '\\'.join(app_path[:-1]) + '\\pred_images\\'
    pred_images = os.listdir(pred_images_path)
    required_images = [i for i in pred_images if sat_image_name in pred_images]

    if (model_type == 'unet_scratch'):
        sp_image = 'pred_noaug_{}'.format(sat_image_name)
    else:
        sp_image = 'pred_aug_{}'.format(sat_image_name)
    
    sp_image_path = pred_images_path + sp_image

    pred_image = cv.imread(sp_image_path)
    pred_image = cv.cvtColor(pred_image, cv.COLOR_BGR2RGB)

    pred_image_fig = px.imshow(pred_image)
    pred_image_fig.update_layout(
        coloraxis_showscale=False,
        autosize=True, height=400,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    pred_image_fig.update_xaxes(showticklabels=False)
    pred_image_fig.update_yaxes(showticklabels=False)

    output_result = html.Div([
        dcc.Graph(id='sat-image', figure=pred_image_fig)
    ])

    return output_result




if __name__ == '__main__':
    app.run_server(debug=True)