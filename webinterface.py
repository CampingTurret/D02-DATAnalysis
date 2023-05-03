
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

@app.route('/getcases')
def get_itemlist():
    validfiles = ['Free.help', 'Locked.help', 'Pre.help', 'Rel0.help', 'Rel50.help', 'Rel100.help']
    AOA = ['0','5']
    F = ['05','5','8','Bend','Flap']
    Plate = ['A','B','B Left','B Right','C']
    q = {}
    for p in Plate:
        for a in AOA:
            for f in F:
                case = f'Plate {p} A{a} F{f}'
                exists = 0
                for v in validfiles:
                    Path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'MODELS', f'Plate {p}', f'Dynamic', f'A{a}', f'F{f}', f'{v}'))
                    if os.path.exists(Path):
                        exists +=1
                if exists==0: q[case] = 'red'
                elif exists<6: q[case] = 'orange'
                elif exists==6:  q[case] = 'green'
    return render_template('cases.html', data=q)               
                    


@app.route('/plot.svg')
def plot_svg():
    return send_from_directory('static', 'plot.svg')


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

                if(request.form.get('Raw') != None):Pa.Plot_Raw_2D_All(selected)
                r2 = Pa.R2_Score(selected)
                Pa.run_analysis_2D_Quick('limit',show=False,types = selected)

            if(request.form.get('limit') == None ):
                if(request.form.get('Raw') != None):Pa.Plot_Raw_2D_All(['Free', 'Locked', 'Pre', 'Rel0', 'Rel50', 'Rel100'])
                r2 = Pa.R2_Score(['Free', 'Locked', 'Pre', 'Rel0', 'Rel50', 'Rel100'])
                Pa.run_analysis_2D_Quick('quick',show=False)

            # Generate image using Matplotlib
            plt.legend()
            plt.xlabel('Time [s]')
            plt.ylabel('Bending moment coefficient [-]')
            plt.savefig(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','static','plot.svg')))
            plt.clf()
        
        if(request.form.get('Train') != None):
            q.put(Pa)

        return render_template('index.html', r2 = r2)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    processes = []
    for _ in range(int(os.cpu_count()/4)+1):
        p = Process(target=trainfunction, args=(q, active_workers))
        p.start()
        processes.append(p)

    app.run(port=2000)

    for p in processes:
        p.join()