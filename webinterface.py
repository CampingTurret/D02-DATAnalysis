
from flask import Flask, render_template, request
from flask import send_from_directory
from multiprocessing import Process, Queue, Value
import matplotlib.pyplot as plt
import matplotlib
import os
from dataloader import data
matplotlib.use('Agg')
app = Flask(__name__)

active_workers = Value('i', 0)
q = Queue()

@app.route('/getqueue')
def get_queue():
    # Get the current items in the queue
    queue_length=q.qsize()
    workers = 0
    for process in processes:
        if process.is_alive:
            workers = workers +1
    Freeworkers = workers - active_workers.value
    # Render the queue.html template and pass the items to it
    return render_template('queue.html', queue_length=queue_length, Active_length = active_workers.value, Free_length = Freeworkers)


@app.route('/plot.png')
def plot_png():
    return send_from_directory('static', 'plot.png')


def trainfunction(q: Queue, active_workers: Value):
    while True:
       
        l = q.get()
        if l is None:
            break
        Pa = l
        with active_workers.get_lock():
            active_workers.value += 1
        Pa.run_Train_2D()
        with active_workers.get_lock():
            active_workers.value -= 1


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from form
        Plate =request.form['input1']
        AOA = request.form['input2']
        prehz = request.form['input3']
        
        if prehz == '0.5': hz = 0.5
        if prehz == '5': hz = 5
        if prehz == '8': hz = 8
        if prehz == 'Flap': hz = 'Flap'
        if prehz == 'Bend': hz = 'Bend'
        
        Pa = data(Plate,AOA,hz,'cpu')
        if(request.form.get('Plot') != None):
            if(request.form.get('limit') != None):
           
                selected = []
                #get user input from form
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
        
        if(request.form.get('Train') != None):
            q.put(Pa)

        return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    processes = []
    for _ in range(1):
        p = Process(target=trainfunction, args=(q, active_workers))
        p.start()
        processes.append(p)

    app.run(port=3000)

    for p in processes:
        p.join()