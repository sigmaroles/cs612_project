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
    # yadda yadda .. matplotlib settings to make it look good
    font = {'family' : 'sans-serif',
        'size'   : 16,
        'weight' : 'bold',
        }
    axes = { 'labelsize': 'x-large' }
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)
   
    # location of trace files w.r.t current dir
    data_dir = 'Trace_files'
    # get the names of all .txt files under trace file directory
    fnames = [x for x in filter(lambda x: x.endswith('.txt'), os.listdir(os.path.join('.', data_dir)))]
    # extract names of workloads, will be useful later
    workload_names = [ x for x in map(lambda x: x.split('_')[1].split('.')[0], fnames) ]

    # create Pandas dataframe from the trace files. 
    # Only read columns 0 and 2
    # dataset is a dict, indexed by strings in workload_names
    # e.g. dataset['radio']
    dataset = {fname.split('_')[1].split('.')[0]:pd.read_csv(os.path.join(data_dir, fname), usecols=[0,2], delimiter='\s+', header = None, dtype =int) for fname in fnames}
        
    # labels for the heatmap axes
    t_ax = [x for x in range(t_MIN,t_MAX,t_STEP)]
    s_ax = [x for x in range(s_MIN, s_MAX, s_STEP)] 
    
    # main loop - for each workload in dataset... 
    for wname in workload_names:
        fig, ax = plt.subplots(figsize=(28,11))
        print ("Beginning with ",wname)
        # df is just a handy shortcut
        df = dataset[wname].values
        # init the empty list to hold the heatmap matrix
        kst = []
        for t in range(t_MIN,t_MAX,t_STEP):
            # init the empty list for storing a row
            kst_row = []
            # get pairs of memory references that are t steps apart in the file
            tlist = np.hstack([df[t:], df[:-t]])
            # the number of references in this list of references (for this t)
            L = len(tlist)
            if not L:
                continue
            # now, given the t list, for each s...
            for s in range(s_MIN, s_MAX, s_STEP):
                # calculate the local/non-local ness of each item in tlist
                local = [isSpatiallyLocal(x, s) for x in tlist]
                # count the matrix value (probability) and store it
                kst_row.append(sum(local) / L)
                #print (kst_row[-1])
            # display progress..it can get quite slow!
            print ("\tt = ", t)
            # save the row to matrix
            kst.append(kst_row)
        
        # plot and save the matrix (heatmap)
        kst = np.array(kst)
        sns.heatmap(kst, ax=ax, robust=True, annot=True, linewidth=0, xticklabels=s_ax, yticklabels=t_ax, center=kst.max())
        ax.set_ylabel('temporal (instructions)', fontsize=18)
        ax.set_xlabel('spatial (bytes)', fontsize=18)
        plt.savefig(wname+'_heatmap_.png')
        plt.clf()
        