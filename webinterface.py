
from flask import Flask, render_template, request
from flask import send_from_directory
import matplotlib.pyplot as plt
import matplotlib
import os
from dataloader import data
matplotlib.use('Agg')
app = Flask(__name__)


@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from form
        Plate =request.form['input1']
        AOA = request.form['input2']
        hz = request.form['input3']
        Pa = data(Plate,AOA,hz,'cpu')
        if(request.form.get('limit') != None):
           
            selected = []
            if (request.form.get('case1') != None): selected.append('Free') 
            if (request.form.get('case2') != None): selected.append('Locked')
            if (request.form.get('case3') != None): selected.append('Pre')
            if (request.form.get('case4') != None): selected.append('Rel0')
            if (request.form.get('case5') != None): selected.append('Rel50')
            if (request.form.get('case6') != None): selected.append('Rel100')

            Pa.run_analysis_2D_Quick('limit',show=False,types = selected)

        if(request.form.get('limit') == None ):
            Pa.run_analysis_2D_Quick('quick',show=False)

        # Generate image using Matplotlib
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Bending [N-mm]')
        plt.savefig(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','static','plot.png')))
        plt.clf()
        

        return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()