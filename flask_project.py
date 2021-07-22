import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, Response
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from matplotlib import pylab as plt
import numpy as np
import nibabel as nib
import atexit
from subprocess import call


app = Flask(__name__)


app.config['ROOT_DIR'] = '/hpf/largeprojects/smiller/users/Katharine/flask_project'
app.config['UPLOAD_EXTENSIONS'] = ['.gz']
app.config['UPLOAD_PATH'] = 'uploaded_images'
app.config['SEGMENT_PATH'] = 'predicted_images'
app.config['MODEL_PATH'] = 'predict/models'

cache = {}
cache['editing'] = True
cache['loaded_image'] = 'none'
cache['loaded_model'] = 'none'
cache['loaded_prediction'] = 'none'
cache['gpu_running'] = False
cache['running_message'] = 0
cache['scale_ranges'] = [140,140,140]
app.config['SCALE_VALUES'] = [50,50,50]
app.config['SEGMENT_TOGGLE'] = False
app.config['fail_safe'] = True

@app.route('/')
def home():
    return render_template('home.html', 
    scale_values = app.config['SCALE_VALUES'], 
    loaded_file = cache['loaded_image'], 
    loaded_model = cache['loaded_model'], 
    loaded_prediction = cache['loaded_prediction'], 
    files = os.listdir(app.config['UPLOAD_PATH']),
    models = os.listdir(app.config['MODEL_PATH']),
    gpu_running_flag = cache['gpu_running'],
    loaded_prediction_path = os.path.join(app.config['UPLOAD_PATH'],cache['loaded_prediction']),
    scale_ranges = cache['scale_ranges'],
    running_message = cache['running_message']
    )

def reload():
    return redirect(url_for('home'))

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if request.method == 'POST': 
   
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)

        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            print(file_ext)
            if file_ext not in app.config['UPLOAD_EXTENSIONS']: #or file_ext != validate_image(uploaded_file.stream):
                abort(400)

            app.config['IMAGE'] = os.path.join(app.config['UPLOAD_PATH'], filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('home'))

@app.route('/upload_models', methods=['POST'])
def upload_models():
    if request.method == 'POST': 
   
        uploaded_file = request.files['model']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            print(file_ext)
            if file_ext not in ['.pt']: #or file_ext != validate_image(uploaded_file.stream):
                abort(400)

            app.config['IMAGE'] = os.path.join(app.config['MODEL_PATH'], filename)
            uploaded_file.save(os.path.join(app.config['MODEL_PATH'], filename))
    return redirect(url_for('home'))


@app.route('/plot.png')
def plot_png():
    fig = create_image()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_image():
    fig = Figure(figsize=(10,2),facecolor='#333B45')  
    file = cache['loaded_image']
    
    if os.path.splitext(file)[1] =='.gz':
        image = np.array(nib.load(os.path.join(app.config['UPLOAD_PATH'],file)).get_fdata(), dtype=np.float32)
        cache['scale_ranges'] = [image.shape[0]-1, image.shape[1]-1, image.shape[2]-1]
        print("Update Scale", cache['scale_ranges'])
        plotAxial(image, fig, 'gray')
        plotSagittal(image, fig, 'gray')
        plotCoronal(image, fig, 'gray')
        
    return fig

def plotAxial(image, fig, cmap):
    print(image.shape)
    coor = min(app.config['SCALE_VALUES'][1],image.shape[1])
    ax = image[:,coor, :]
    ax = np.rot90(ax,axes=(-2,-1)) 
    p = fig.add_subplot(132)
    p.imshow(ax, cmap=cmap)
    # plot = p.pcolor(ax)
    # fig.colorbar(plot)


def plotSagittal(image, fig, cmap):
    coor = min(app.config['SCALE_VALUES'][0],image.shape[0])
    sg = image[coor, :, :]
    sg = np.rot90(sg,axes=(-2,-1)) 
    p = fig.add_subplot(131)
    p.imshow(sg, cmap=cmap)

def plotCoronal(image, fig, cmap):
    coor = min(app.config['SCALE_VALUES'][2],image.shape[2])
    cr = image[:, :, coor]
    cr = np.rot90(cr,axes=(-2,-1)) 
    p = fig.add_subplot(133)
    p.imshow(cr, cmap=cmap)

@app.route("/update_slider", methods=["POST"])
def update_slider():
    app.config['SCALE_VALUES'][0] = int(request.form["sagittal"])
    app.config['SCALE_VALUES'][1] = int(request.form["axial"])
    app.config['SCALE_VALUES'][2] = int(request.form["coronal"])
    return redirect(url_for('home'))


@app.route("/connect_gpu", methods=["POST"])
def connect_gpu():
    if not cache['gpu_running']:
        os.system('tmux send-keys -t 1 qsub Space -q Space gpu Space -l Space nodes=1:ppn=1:gpus=1,mem=20g,walltime=12:00:00 Space -I Enter')
        os.system('tmux send-keys -t 1 cd Space '+app.config['ROOT_DIR']+' Enter')
        cache['gpu_running'] = True
    return redirect(url_for('home'))

@app.route("/close_gpu", methods=["POST"])
def close_gpu():
    if cache['gpu_running']:
        print('closing gpu session')
        os.system('tmux send-keys -t 1 C-c')
        os.system('tmux send-keys -t 1 C-d')
        cache['gpu_running'] = False
    return redirect(url_for('home'))

@app.route("/load_prediction", methods=["POST"])
def load_prediction():
    cache['running_message'] = 0
    model_name_short = cache['loaded_model'].split('.')[0]
    expected_name = model_name_short+'_'+cache['loaded_image']
    print(expected_name)
    print(cache['loaded_image'])
    if expected_name in os.listdir(app.config['SEGMENT_PATH']):
        cache['loaded_prediction'] = expected_name
    else:
        temp_file = open("./predict/predicted_path.txt","r+")
        prediction_file = temp_file.readline()
        temp_file.truncate(0)
        temp_file.close()
        if prediction_file != '':
            cache['loaded_prediction'] = prediction_file
        else:
            cache['loaded_prediction'] = 'none'
    print("HERE")
    print("AHH " + cache['loaded_image']+" HELLO "+cache['loaded_model'])
    return redirect(url_for('home'))

@app.route("/predict_button", methods=["POST"])
def predict_button():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("DEVICE", device)
    if cache['loaded_image'] != '' and cache['loaded_model'] != '' and cache['loaded_image'] != 'none'and cache['loaded_model'] != 'none':
        print("STARTING", cache['loaded_image'])
        temp_file = open("./predict/image_path.txt","w")
        temp_file.write(cache['loaded_image']+'\n'+cache['loaded_model'])
        temp_file.close()
        
        result = os.system('tmux send-keys -t 1 . Space ./run/runPython.sh Enter')
        #result = os.system('tmux send-keys -t 1 qsub Space -q Space gpu Space -l Space nodes=1:ppn=1:gpus=1,vmem=20g,mem=20g,walltime=12:00:00 Space run/runPython.sh Enter')
        
    
        # finished = False
        # while not finished:
        #     print("running")
        #     cache['running_message'] = 1
        #     temp_file = open("./predict/predicted_path.txt","r")
        #     prediction_file = temp_file.readline()
        #     temp_file.close()
        #     if  cache['loaded_image'] and cache['loaded_model'] in prediction_file: finished = True
        # print("Prediction ready. GPU node may still be running predict script.")
        # cache['running_message'] = 2
        # # temp_file = open("./predict/predicted_path.txt","r+")
        # # prediction_file = temp_file.readline()
        # # temp_file.truncate(0)
        # # temp_file.close()
        # # if prediction_file != '':
        # #     cache['loaded_prediction'] = prediction_file
        # # else:
        # #     cache['loaded_prediction'] = 'none'



    return redirect(url_for('home'))
    
@app.route("/load_file", methods=["POST"])
def load_file():
    selected = request.form.get("file_picker")
    if 'nii.gz' in selected and request.form.get("file_picker")!='':
        cache['loaded_image'] = request.form.get("file_picker")
        print('LOADED', cache['loaded_image'])
    return redirect(url_for('home'))

@app.route("/load_model", methods=["POST"])
def load_model():
    selected = request.form.get("model_picker")
    if '.pt' in selected and request.form.get("model_picker")!='':
        cache['loaded_model'] = request.form.get("model_picker")
        print('MODEL', cache['loaded_model'])
    return redirect(url_for('home'))



@app.route('/segment.png')
def segment_png():
    fig = create_segment()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_segment():
    # cache['gpu_running'] = True
    # print(cache['gpu_running'])
    fig = Figure(figsize=(10,2),facecolor='#333B45')
    file = cache['loaded_prediction']
    
    if os.path.splitext(file)[1] =='.gz':
        image = np.array(nib.load(os.path.join(app.config['SEGMENT_PATH'],file)).get_fdata(), dtype=np.float32)
        # cache['scale_ranges'] = image.shape
        # print("Update Scale", cache['scale_ranges'])
        plotAxial(image, fig, 'cubehelix')
        plotSagittal(image, fig, 'cubehelix')
        plotCoronal(image, fig, 'cubehelix')

    return fig

@app.route('/predicted_images/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.config['ROOT_DIR'],app.config['SEGMENT_PATH'])
    print("HI", uploads)
    #os.path.join(app.config['UPLOAD_PATH'],cache['loaded_prediction'])
    return send_from_directory(directory=uploads, path=filename)

def exit_handler():
    print('End Sequence')
    if cache['gpu_running'] and not cache['editing']:
        print('closing gpu session')
        os.system('tmux send-keys -t 1 C-d')
        cache['gpu_running'] = False

atexit.register(exit_handler)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8088)
