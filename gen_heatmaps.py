import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns

# parameters of the probability function
s_MAX = 2**14
s_MIN = 2**11
s_STEP = 2**11
t_MAX = 1000
t_MIN = 30
t_STEP = 85


"""
requirements to be "local" in terms of s:
* instructions are within s distance of each other
* the request windows overlap (n1 and n2)
"""
    
def isSpatiallyLocal(row, s):
    a1, n1, a2, n2 = row
    if a1<=a2:
        return (a2<=a1+s) and (a1+n1>=a2)
    else:
        return (a1<=a2+s) and (a2+n2>=a1)
    
if __name__=='__main__':
    font = {'family' : 'sans-serif',
        'size'   : 16,
        'weight' : 'bold'
        }
    matplotlib.rc('font', **font)
    #plt.rcParams["figure.figsize"] = [26, 13]
    
    data_dir = 'Trace_files'
    fnames = [x for x in filter(lambda x: x.endswith('.txt'), os.listdir(os.path.join('.', data_dir)))]
    workload_names = [ x for x in map(lambda x: x.split('_')[1].split('.')[0], fnames) ]

    col_names = ['a', 'n']
    dataset = {fname.split('_')[1].split('.')[0]:pd.read_csv(os.path.join(data_dir, fname), usecols=[0,2], delimiter='\s+', header = None, dtype =int) for fname in fnames}
        
    t_ax = [x for x in range(t_MIN,t_MAX,t_STEP)]
    s_ax = [x for x in range(s_MIN, s_MAX, s_STEP)] 
    
    # main loop - for each workload in dataset...
    for wname in workload_names:
        fig, ax = plt.subplots(figsize=(28,11))
        print ("Beginning with ",wname)
        #data = [tuple(x) for x in dataset[wname].values]
        df = dataset[wname].values
        kst = []
        for t in range(t_MIN,t_MAX,t_STEP):
            # init the container for storing the row for this t
            kst_row = []
            # get pairs of memory references that are t steps apart in the file
            tlist = np.hstack([df[t:], df[:-t]])
            L = len(tlist)
            for s in range(s_MIN, s_MAX, s_STEP):
                # a list of booleans, indicating locality of each pair in tlist
                local = [isSpatiallyLocal(x, s) for x in tlist]
                kst_row.append(sum(local) / (L - t))
            print ("\tt = ", t)
            kst.append(kst_row)
        kst = np.array(kst)
        sns.heatmap(kst, ax=ax, robust=True, annot=True, linewidth=0, xticklabels=s_ax, yticklabels=t_ax, center=kst.max())
        ax.set_ylabel('temporal (instructions)')
        ax.set_xlabel('spatial (bytes)')
        plt.savefig(wname+'_heatmap_.png')
        plt.clf()
        