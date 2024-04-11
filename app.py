from shiny import ui, render, App, reactive, Inputs, Outputs, Session
from htmltools import br, h5
from shiny.types import ImgData

import evaluate as evl
import create_results as res
from data_set_params import DataSetParams
import classifier as clss
import run_detector as rd

import numpy as np
import pandas as pd
import pickle5 as pickle
import os


#run_comparison
test_set = 'bulgaria'  # can be one of: bulgaria, uk, norfolk, all
data_set = 'data/train_test_split/test_set_' + test_set + '.npz'
raw_audio_dir = 'data/wav/'
result_dir = 'results/'
model_dir = 'data/models/'

# train and test_pos are in units of seconds
loaded_data_tr = np.load(data_set, allow_pickle=True, encoding='latin1') #allow_pickle=True, encoding='latin1'
train_pos = loaded_data_tr['train_pos']
train_files = loaded_data_tr['train_files']
train_durations = loaded_data_tr['train_durations']
test_pos = loaded_data_tr['test_pos']
test_files = loaded_data_tr['test_files']
test_durations = loaded_data_tr['test_durations']

# load parameters
params = DataSetParams()
params.audio_dir = raw_audio_dir

############################
def cnn_train_and_test ():
    params.classification_model = 'cnn'
    model = clss.Classifier(params)
    # train and test
    model.train(train_files, train_pos, train_durations)
    nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
    # compute precision recall
    precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
    res.plot_prec_recall('cnn', recall, precision, result_dir, nms_prob)
    #plot epochs
    #res.plot_epochs(history = training_history)
    # save CNN model to file
    pickle.dump(model, open(model_dir + 'test_set_' + test_set + '.mod', 'wb'))
    #return True
##############################

#run_detector
# params
detection_thresh = 0.80   # make this smaller if you want more calls detected
do_time_expansion = True  # set to True if audio is not already time expanded
save_res = True

data_dir = 'Rufe/'   #path of the data that we run the model on
op_ann_dir = 'results/'    # where we will store the outputs
op_file_name_total = op_ann_dir + 'op_file.csv'

model_file = model_dir + 'test_set_bulgaria.mod'


app_ui = ui.page_fluid(

    ui.panel_title("Bat Detector"),
    ui.navset_tab(
        ui.nav("Train", 
            ui.layout_sidebar(
                ui.sidebar(
                    ui.navset_tab(
                        ui.nav("Settings",
                            h5("CNN Parameters"),
                            ui.input_numeric("learn_rate", "Learn rate", 0.01, step=0.01, min=0),
                            ui.input_numeric("moment", "Moment", 0.9, step=0.1, min=0),
                            ui.input_numeric("epochs", "Number of Epochs", 50, step=1, min=0),
                            ui.input_numeric("batchsize", "Batchsize", 256, step=1, min=0),
                            ui.div({"style": "display: none;"}, ui.input_text("loading_train", "loading_train")),
                            ui.input_action_button("params_entry", "Update"),
                            br(), br(),
                            ui.input_action_button("train_and_test", "Start"),
                        ),
                    ),     
                    id="sidebar_left", width="30%"
                ),
                ui.navset_tab(
                    ui.nav("Info",
                        ui.output_text("update_text"),
                        ui.output_text("cnn_params_text"),
                        br(), br(),
                        ui.output_text("train_start"),
                        ui.output_text("train_end"),
                    ),
                    ui.nav("Results",
                        ui.output_image("plot_prec_recall"),
                    ),
                ),
            ),
        ),
        ui.nav("Detect",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_radio_buttons( 
                        "model_choice", 
                        "Choice of CNN-Model", 
                        {"1": "Own", "2": "Pre-trained"},
                    ),
                    ui.input_file("wav_files", "Choose WAV Files", accept=[".wav"], multiple=True),
                    ui.input_numeric("detection_thresh", "Detection threshold", 0.8, step=0.1, min=0.5),
                    ui.div({"style": "display: none;"}, ui.input_text("loading_detect", "loading_detect")),
                    ui.input_action_button("save_detect", "Save"),
                    ui.input_action_button("start_detect", "Start"), 
                    id="sidebar_left", width="30%"
                ),
                ui.navset_tab(
                    ui.nav("Info",
                        ui.output_text("model_choice_text"),
                        ui.output_text("detection_thresh_text"),
                        ui.output_text("wav_data_dir_text"),
                        br(), br(),
                        ui.output_text("start_detection_text"),
                        ui.output_text("end_detection_text"),  
                    ),
                    ui.nav("Results",
                        ui.output_table("result"),
                    ),
                ),
            ),
        ),
    )
)

@reactive.file_reader(op_file_name_total)
def read_op_file():
    return pd.read_csv(op_file_name_total)

def server(input: Inputs, output: Outputs, session: Session):

    @output
    @render.text
    @reactive.event(input.params_entry)
    def update_text():
        params.learn_rate = input.learn_rate()
        params.moment = input.moment()
        params.num_epochs = input.epochs()
        params.batchsize = input.batchsize()
        return "Updated variables:" 
    
    @output
    @render.text
    @reactive.event(input.params_entry)
    def cnn_params_text():
        cpt = "Learn rate = {} -- Moment = {} -- Epochs = {} -- Batchsize = {}"
        return cpt.format(params.learn_rate, params.moment, params.num_epochs, params.batchsize)
    
    @reactive.Effect
    @reactive.event(input.train_and_test)
    def _():
        ui.update_text("loading_train", value=1)

    @reactive.Effect
    def _():
        if input.loading_train() == "2":
            cnn_train_and_test()
            ui.update_text("loading_train", value=0)

    @render.text
    @reactive.event(input.loading_train)
    def train_start():
        if input.loading_train() == "1":
            ui.update_text("loading_train", value=2)
            return "Model is getting trained..."

    @render.text
    @reactive.event(input.loading_train)
    def train_end():
        if input.loading_train() == "0":
            return "It is ready!!!"
        
    @render.image
    def plot_prec_recall():
        dir = result_dir + "plot_prec_recall.png"
        img: ImgData = {"src": dir}
        return img

    @output
    @render.text
    @reactive.event(input.save_detect)
    def model_choice_text():
        global model_file
        global model_dir
        if(input.model_choice () == "1"):
            model_file = model_dir + 'test_set_' + test_set + '.mod'
            return "own model selected"
        elif(input.model_choice () == "2"):
            model_dir = 'data/models/pre-trained/'
            model_file = model_dir + 'test_set_bulgaria.mod'
            return "pre-trained model selected"
    
    @output
    @render.text
    @reactive.event(input.save_detect)    
    def wav_data_dir_text ():
        data_dir_infos = input.wav_files ()
        
        for file_info in data_dir_infos:
            file_path = file_info["datapath"]
            false_name = os.path.basename(file_path)
            real_name = file_info["name"]
            corrected_file_path = file_path.replace(false_name, real_name)
            file_info["datapath"] = corrected_file_path
        
        if not data_dir_infos:
           return "there is no file choosen"
        else:
            my_string = data_dir_infos[0]["datapath"]
            words = my_string.split('/')
            if len(words) > 1:
                res = '/'.join(words[:-1])
            else:
                res = ""
            global data_dir
            data_dir = res + "/"
            return "Files are saved."
    
    @output
    @render.text
    @reactive.event(input.save_detect)
    def detection_thresh_text ():
        detection_thresh = input.detection_thresh ()
        return "Detection threshold set to be " + str(detection_thresh)  

    @reactive.Effect
    @reactive.event(input.start_detect)
    def _():
        ui.update_text("loading_detect", value=1)

    @reactive.Effect
    async def _():
        if input.loading_detect() == "2":
            await rd.main(detection_thresh, do_time_expansion, save_res, data_dir, op_ann_dir, op_file_name_total, model_file)
            ui.update_text("loading_detect", value=0)

    @render.text
    @reactive.event(input.loading_detect)
    def start_detection_text():
        if input.loading_detect() == "1":
            ui.update_text("loading_detect", value=2)
            return "Detection of bat passes is proceeding..."

    @render.text
    @reactive.event(input.loading_detect)
    def end_detection_text():
        if input.loading_detect() == "0":
            return "Detection is done"

    @render.table
    def result():
        return read_op_file()


app = App(app_ui, server)