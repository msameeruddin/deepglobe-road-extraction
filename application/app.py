import warnings
warnings.filterwarnings('ignore')

import os
import dash
import time
import base64
import dash_daq as daq
import numpy as np
import cv2 as cv
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt

from tensorflow.keras import Input as TFInput
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Activation,
    Concatenate
)
from tensorflow.keras.models import Model

from dash import html, dcc
from dash.dependencies import (
    Input, Output, State
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


## reading the encoded (string) image into matrix
def read_image_string(contents):
   encoded_data = contents[0].split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   image = cv.imdecode(nparr, cv.IMREAD_COLOR)
   image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
   return image

def parse_contents(contents, filenames, dates):
    image_mat = read_image_string(contents=contents)
    return image_mat

## UNet model
## Reference → https://youtu.be/GAYJ81M58y8
## The below code is taken from the above video link. Although I have modified it.
## I give full credit to the author of the video and the creators of UNet.

class UNET:
    # convolution block
    def _convolve(self, input_, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    # up-sampling and convolotion block
    def _convolve_by_upsampling(self, input_, skip_connector, filters, rate):
        x = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_)
        x = concatenate([x, skip_connector])
        x = Dropout(rate)(x)
        x = self._convolve(input_=x, filters=filters)
        return x
    
    # UNET main model 
    def unet_main(self, input_, filters=16, rate=0.05):
        # left encoder Path
        c1 = self._convolve(input_=input_, filters=filters)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        p1 = Dropout(rate)(p1)
        
        c2 = self._convolve(input_=p1, filters=filters * 2)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        p2 = Dropout(rate)(p2)
        
        c3 = self._convolve(input_=p2, filters=filters * 4)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        p3 = Dropout(rate)(p3)
        
        c4 = self._convolve(input_=p3, filters=filters * 8)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(rate)(p4)

        # middle bridge
        c5 = self._convolve(input_=p4, filters=filters * 16)
        
        # right decoder path
        c6 = self._convolve_by_upsampling(input_=c5, skip_connector=c4, filters=filters * 8, rate=rate)
        c7 = self._convolve_by_upsampling(input_=c6, skip_connector=c3, filters=filters * 4, rate=rate)
        c8 = self._convolve_by_upsampling(input_=c7, skip_connector=c2, filters=filters * 2, rate=rate)
        c9 = self._convolve_by_upsampling(input_=c8, skip_connector=c1, filters=filters * 1, rate=rate)
        
        output_ = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs=[input_], outputs=[output_])
        
        return model


## layout design
app.layout = html.Div([
    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),

    html.Div(
        id='title-app', 
        children=[html.H3(app.title)],
        style={'textAlign' : 'center', 'paddingTop' : 30}
    ),

    html.Div([
        html.Div([

            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    multiple=True
                ),
            ], style={
                    'height': '70px',
                    'lineHeight': '50px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'backgroundColor': '#F0F1F1'
                }
            ),

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
            ], style={'textAlign' : 'center', 'paddingTop' : 70}),

            html.Div([
                daq.ToggleSwitch(
                    id='extraction-mode',
                    size=60,
                    label='Extract Path',
                    labelPosition='top',
                    color='red',
                    value=False,
                )
            ], style={'paddingTop' : 70})

        ], className='four columns'),

        html.Div([
            html.Div(id='result-in-out-image'),
        ], className='eight columns')
    
    ], className='row', style={'paddingTop' : 50}),

], className='container')


## fetch the model weights
def get_unet_model(model_name):
    u = UNET()
    
    inputs = TFInput(shape=(256, 256, 3))
    model = u.unet_main(input_=inputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    project_path = '/'.join(os.getcwd().split('/')[:-1])
    model_path = project_path + '/models/{}.h5'.format(model_name)

    if not os.path.isfile(path=model_path):
        print('Please train the model first.')
    else:
        model.load_weights(model_path)
    
    return model

## callbacks
@app.callback(
    Output('result-in-out-image', 'children'), 
    [
        Input('upload-image', 'contents'),
        Input('model-dropdown', 'value'),
        Input('extraction-mode', 'value'),
        # -------
        State('upload-image', 'filename'),
        State('upload-image', 'last_modified'),
    ]
)
def display_image(contents, model_name, is_on, filenames, dates):
    if contents is not None:
        sat_image = parse_contents(contents, filenames, dates)
        simage_shape = sat_image.shape

        if not is_on:
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

        
        model = get_unet_model(model_name=model_name)
        # prediction
        predicted_image  = model.predict(sat_image[np.newaxis,:,:,:])
        predicted_mask = predicted_image.reshape(simage_shape[0], simage_shape[1])

        # automatic threshold identification
        pmask_thresh = threshold_otsu(predicted_mask)
        # binarizing the mask
        th, predicted_mask = cv.threshold(src=predicted_mask, thresh=pmask_thresh, maxval=255, type=cv.THRESH_BINARY)

        rs, gs, bs = cv.split(sat_image)
        rsp = rs + predicted_mask
        rsp = np.where((rsp > 255), 255, rsp)

        gsp = gs + predicted_mask
        gsp = np.where((gsp > 255), 0, gsp)

        bsp = bs + predicted_mask
        bsp = np.where((bsp > 255), 0, bsp)

        out_image = np.dstack((rsp, gsp, bsp))
        out_image = out_image.astype('uint8')

        out_image_fig = px.imshow(out_image)
        out_image_fig.update_layout(
            coloraxis_showscale=False,
            autosize=True, height=400,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)

        output_result = html.Div([
            dcc.Graph(id='sat-image', figure=out_image_fig)
        ])

        return output_result












if __name__ == '__main__':
    app.run_server(debug=True)