import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import imageio
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import dash
from dash import Dash, Input, Output, dcc, html, State
from dash.exceptions import PreventUpdate
from .self_prompting import grounding_dino_prompt
from google.cloud import speech

def mark_image(_img, points):
    assert(len(points) > 0)
    img = _img.copy()
    r = 10
    mark_color = np.array([255, 0, 0]).reshape(1, 1, 3)
    for i in range(len(points)):
        point = points[i]
        img[point[1]-r:point[1]+r+1, point[0]-r:point[0]+r+1] = mark_color
    return img

def draw_figure(fig, title, animation_frame=None):
    fig = px.imshow(fig, animation_frame=animation_frame)
    if animation_frame is not None:
        # fig.update_layout(sliders = [{'visible': False}])
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 33
    fig.update_layout(title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


class Sam3dGUI:
    def __init__(self, Seg3d, debug=False):
        ctx = {
            'num_clicks': 0, 
            'click': [], 
            'cur_img': None, 
            'btn_clear': 0, 
            'btn_text': 0, 
            'btn_audio': 0,
            'prompt_type': 'point',
            'show_rgb': False
            }
        self.ctx = ctx
        self.Seg3d = Seg3d
        self.debug = debug

        self.train_idx = 0

    def run(self):
        init_rgb = self.Seg3d.init_model()
        self.ctx['cur_img'] = init_rgb
        self.run_app(sam_pred=self.Seg3d.predictor, ctx=self.ctx, init_rgb=init_rgb)


    def run_app(self, sam_pred, ctx, init_rgb):
        '''
        run dash app
        '''
        def send_file_to_google(upload_file_name):
            return transcribe_model_selection(speech_file=upload_file_name, model="latest_short")
        
        def transcribe_model_selection(
            speech_file: str,
            model: str,) -> speech.RecognizeResponse:
            """Transcribe the given audio file synchronously with
            the selected model."""
            client = speech.SpeechClient()

            with open(speech_file, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                language_code="en-US",
                model=model,
            )

            response = client.recognize(config=config, audio=audio)

            for i, result in enumerate(response.results):
                alternative = result.alternatives[0]
                print("-" * 20)
                print(f"First alternative of result {i}")
                print(f"Transcript: {alternative.transcript}")

            return alternative.transcript
        
        def query(points=None, text=None):
            with torch.no_grad():
                if text is None:
                    input_point = points
                    input_label = np.ones(len(input_point))
                    masks, scores, logits = sam_pred.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                elif points is None:
                    input_boxes = grounding_dino_prompt(ctx['cur_img'], text)
                    boxes = torch.tensor(input_boxes)[0:1].cuda()
                    transformed_boxes = sam_pred.transform.apply_boxes_torch(boxes, ctx['cur_img'].shape[:2])
                    masks, scores, logits = sam_pred.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=True,
                    )
                    masks = masks[0].cpu().numpy()
                else:
                    raise NotImplementedError

            fig1 = (255*masks[0, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig2 = (255*masks[1, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig3 = (255*masks[2, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig1 = draw_figure(fig1, 'mask0')
            fig2 = draw_figure(fig2, 'mask1')
            fig3 = draw_figure(fig3, 'mask2')

            if text is None:
                fig0 = mark_image(ctx['cur_img'], points)
            else:
                fig0 = ctx['cur_img']
            fig0 = draw_figure(fig0, 'original_image')

            return  masks, fig0, fig1, fig2, fig3
        
        # _, fig0, fig1, fig2, fig3, desc = query(np.array([[100, 100], [101, 101]]))
        self.ctx['fig0'] = draw_figure(init_rgb, 'original_image')
        self.ctx['fig1'] = draw_figure(np.zeros_like(init_rgb), 'mask0')
        self.ctx['fig2'] = draw_figure(np.zeros_like(init_rgb), 'mask1')
        self.ctx['fig3'] = draw_figure(np.zeros_like(init_rgb), 'mask2')
        self.ctx['fig_seg_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Masked image in Training')
        self.ctx['fig_sam_mask'] = draw_figure(np.zeros_like(init_rgb), 'SAM Mask with Prompts in Training')
        self.ctx['fig_masked_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Masked RGB')
        self.ctx['fig_seged_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Seged RGB')
        
        app = dash.Dash(
            __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
        )
        app.layout = html.Div(
            style={"height": "100%"},
            children=[
            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SAM Init'])]),
                        html.Br(),

                        html.H5('Prompt Type:'),
                        html.Div([
                            dcc.Dropdown(
                                id = 'prompt_type',
                                options = [{'label': 'Points', 'value': 'point'}, 
                                        {'label': 'Text', 'value': 'text'},
                                        {'label': 'Audio', 'value': 'audio'}],
                                value = 'point'),
                                html.Div(id = 'output-prompt_type')
                        ]),
                        html.Br(),

                        html.H5('Point Prompts:'),
                        html.Button('Clear Points', id='btn-nclicks-clear', n_clicks=0),
                        html.Br(),

                        html.H5('Text Prompt:'),
                        html.Div([
                            dcc.Input(id='input-text-state', type='text', value='none'),
                            html.Button(id='submit-button-state', n_clicks=0, children='Generate'),
                            html.Div(id='output-state-text')
                        ]),
                        html.Br(),

                        # html.H5('Audio File Upload'),
                        # html.Div([
                        #     dcc.Input(id='input-file', type='file', value='filename'),
                        #     dcc.Upload()
                        #     html.Button(id='file-submit-button-state', n_clicks=0, children='Upload and Generate'),
                        #     html.Div(id='output-file')
                        # ]),
                        # html.Br(),

                         dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '85%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            max_size=-1,
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                        html.Div(id='output-data-upload'),
                        html.Button('Process Audio File', id='submit-audio-button-state', n_clicks=0),
                        html.Br(),

                        html.H5('Please select the mask:'),
                        html.Div([
                            dcc.RadioItems(['mask0', 'mask1', 'mask2'], id='sel_mask_id', value=None)
                        ], style={'display': 'flex'}),
                        html.Br(),

                        html.H5(id='container-sel-mask'),
                    ]),
                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='main_image', figure=self.ctx['fig0'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask0', figure=self.ctx['fig1'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask1', figure=self.ctx['fig2'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask2', figure=self.ctx['fig3'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ])
                ])
            ]),

            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SA3D Training'])]),
                        html.Br(),

                        html.Button('Start Training', id='btn-nclicks-training', n_clicks=0),
                        html.Div(id='container-button-training', style={'display': 'inline-block'}),
                        ]),

                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='seg_rgb', figure=self.ctx['fig_seg_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='sam_mask', figure=self.ctx['fig_sam_mask'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ]),

                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000,  # in milliseconds
                        n_intervals=0),
                ])
            ]),

            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SA3D Rendering Results'])]),
                        html.Br(),
                        ]),

                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='masked_rgb', figure=self.ctx['fig_masked_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='seged_rgb', figure=self.ctx['fig_seged_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ]),
                ])
            ])
            
        ])

        @app.callback(Output('output-prompt_type', 'children'), [Input('prompt_type', 'value')])
        def update_prompt_type(value):
            self.ctx['prompt_type'] = value
            if value != 'point':
                ctx['click'] = []
                ctx['num_clicks'] = 0
            return f"Type {value} is chosen"
        
        # @app.callback(dash.dependencies.Output('output-data-upload', 'children'),
        #      [dash.dependencies.Input('upload-data', 'contents'),
        #       dash.dependencies.Input('upload-data', 'filename')])
        # def update_output(contents, filename):
        #     if contents is not None:
        #         # do something with contents
        #         children='processed data from file ' + filename
        #         return children
        #     else:
        #         return 'no contents'
        

        @app.callback(
            Output('main_image', 'figure'),
            Output('mask0', 'figure'),
            Output('mask1', 'figure'),
            Output('mask2', 'figure'),
            Output('output-state-text', 'children'),
            Input('main_image', 'clickData'),
            Input('btn-nclicks-clear', 'n_clicks'),
            Input('submit-button-state', 'n_clicks'),
            State('input-text-state', 'value'),
            Input('submit-audio-button-state', 'n_clicks'),
            # Output('output-data-upload', 'children'),
            Input('upload-data', 'contents'),
            Input('upload-data', 'filename')
        )
        def update_prompt(clickData, btn_point, btn_text, text, btn_audio, upload_data, upload_file_name):
            '''
            update mask
            '''
            if self.ctx['prompt_type'] == 'point':
                if clickData is None and btn_point == self.ctx['btn_clear']:
                    raise PreventUpdate

                if btn_point > self.ctx['btn_clear']:
                    self.ctx['btn_clear'] += 1
                    ctx['click'] = []
                    ctx['num_clicks'] = 0
                    return self.ctx['fig0'], self.ctx['fig1'], self.ctx['fig2'], self.ctx['fig3'], 'none'
                
                ctx['num_clicks'] += 1
                ctx['click'].append(np.array([clickData['points'][0]['x'], clickData['points'][0]['y']]))
                
                ctx['saved_click'] = np.stack(ctx['click'])
                masks, fig0, fig1, fig2, fig3 = query(ctx['saved_click'])
                ctx['masks'] = masks
                return fig0, fig1, fig2, fig3, 'none'
            
            elif self.ctx['prompt_type'] == 'text':
                if btn_text > self.ctx['btn_text']:
                    self.ctx['btn_text'] += 1
                    self.ctx['text'] = text
                    masks, fig0, fig1, fig2, fig3 = query(points=None, text=text)
                    ctx['masks'] = masks
                    return fig0, fig1, fig2, fig3, u'''
                        Input text is "{}"
                    '''.format(text)
                else:
                    raise PreventUpdate
                
            elif self.ctx['prompt_type'] == 'audio':
                if btn_audio > self.ctx['btn_audio']:
                    self.ctx['btn_audio'] += 1
                    print(f"upload file name is: {upload_file_name}")
                    text = send_file_to_google(upload_file_name)
                    self.ctx['text'] = text
                    masks, fig0, fig1, fig2, fig3 = query(points=None, text=text)
                    ctx['masks'] = masks
                    return fig0, fig1, fig2, fig3, u'''
                        Input audio parsed as text is "{}"
                    '''.format(text)
                else:
                    raise PreventUpdate
            else:
                raise NotImplementedError

        @app.callback(
            Output("container-sel-mask", 'children'),
            Input("sel_mask_id", 'value')
        )
        def update_graph(radio_items):
            if radio_items == 'mask0':
                ctx['select_mask_id'] = 0
                return html.Div("you select mask0")
            elif radio_items == 'mask1':
                ctx['select_mask_id'] = 1
                return html.Div("you select mask1")
            elif radio_items == 'mask2':
                ctx['select_mask_id'] = 2
                return html.Div("you select mask2")
            else:
                raise PreventUpdate
            
        @app.callback(
            Output('seg_rgb', 'figure'),
            Output('sam_mask', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def displaySeg(n):
            if self.ctx['show_rgb']:
                self.ctx['show_rgb'] = False
                fig_seg_rgb = draw_figure(self.ctx['fig_seg_rgb'], 'Masked image in Training')
                fig_sam_mask = draw_figure(self.ctx['fig_sam_mask'], 'SAM Mask with Prompts in Training')
                return fig_seg_rgb, fig_sam_mask
            else:
                raise PreventUpdate

        @app.callback(
            Output('container-button-training', 'children'),
            Output('masked_rgb', 'figure'),
            Output('seged_rgb', 'figure'),
            Input('btn-nclicks-training', 'n_clicks')
        )
        def start_training(btn):
            if btn < 1:
                return html.Div("Press to start training"), self.ctx['fig_masked_rgb'], self.ctx['fig_seged_rgb']
            else:
                # optim in the first view
                self.Seg3d.train_step(self.train_idx, sam_mask=ctx['masks'][ctx['select_mask_id']])
                self.train_idx += 1

                # cross-view training
                while True:
                    rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
                    self.train_idx += 1
                    self.ctx['fig_seg_rgb'] = rgb
                    self.ctx['fig_sam_mask'] = sam_prompt
                    self.ctx['show_rgb'] = True
                    if is_finished:
                        break
                self.Seg3d.save_ckpt()
                masked_rgb, seged_rgb = self.Seg3d.render_test()
                fig_masked_rgb = draw_figure(masked_rgb, 'Masked RGB', animation_frame=0)
                fig_seged_rgb = draw_figure(seged_rgb, 'Seged RGB', animation_frame=0)

                return html.Div("Train Stage Finished! Press Ctrl+C to Exit!"), fig_masked_rgb, fig_seged_rgb
            
        
        app.run_server(debug=self.debug)

if __name__ == '__main__':
    from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
    class Sam_predictor():
        def __init__(self, device):
            sam_checkpoint = "./dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
            self.predictor = SamPredictor(self.sam)
            print('sam inited!')
            # pass

        def forward(self, points, multimask_output=True, return_logits=False):
            # self.predictor.set_image(image)
            # input_point = np.array([[x, y], [x + 1, y + 1]]) # TODO, add interactive mode
            input_point = points
            input_label = np.ones(len(input_point))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=multimask_output,
                return_logits=return_logits
            )
            return masks
        
    image = cv2.cvtColor(cv2.imread('data/nerf_llff_data/fern/images_4/image000.png'), cv2.COLOR_BGR2RGB)
    sam_pred = Sam_predictor(torch.device('cuda'))
    sam_pred.predictor.set_image(image)
    video = np.stack(imageio.mimread('logs/llff/fern/render_train_coarse_segmentation_gui/video.rgbseg_gui.mp4'))
    gui = Sam3dGUI(None, debug=True)
    gui.ctx['cur_img'] = image
    gui.ctx['video'] = video
    gui.run_app(sam_pred.predictor, gui.ctx, image)

