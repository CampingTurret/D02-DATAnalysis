
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
        limit = request.form.get('limit')
        print(request.form.get('limit') != None)
        Pa = data(Plate,AOA,hz,'cpu')
        if(int(limit) ==1 ):
           
            selected = []
            if (request.form.get('case1') != None): selected.append('Free') 
            if (request.form.get('case2') != None): selected.append('Locked')
            if (int(request.form.get('case3')) == 1): selected.append('Pre')
            if (int(request.form.get('case4')) == 1): selected.append('Rel0')
            if (int(request.form.get('case5')) == 1): selected.append('Rel50')
            if (int(request.form.get('case6')) == 1): selected.append('Rel100')

            Pa.run_analysis_2D_Quick('limit',show=False,types = selected)

        if(limit !=1 ):
            Pa.run_analysis_2D_Quick('quick',show=False)

        # Generate image using Matplotlib
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Bending [N-mm]')
        plt.savefig(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','static','plot.png')))
        plt.clf()
        plot_url = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','static','plot.png'))
        

        return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()