
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
from datetime import datetime, timedelta


# In[18]:


font = {'family' : 'sans-serif',
        'size'   : 10}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = [19, 9]
#plt.rcParams[""] = False


# In[3]:


data_dir = 'Trace_files'
fnames = [x for x in filter(lambda x: x.endswith('.txt'), os.listdir(os.path.join('.', data_dir)))]
workload_names = [ x for x in map(lambda x: x.split('_')[1].split('.')[0], fnames) ]


# In[4]:


col_names = ['a', 'n']
dataset = {fname.split('_')[1].split('.')[0]:pd.read_csv(os.path.join(data_dir, fname), usecols=[0,2], delimiter='\s+', header = None, dtype =int) for fname in fnames}
for wname in workload_names:
    dataset[wname].columns = col_names


# In[5]:


dataset['music'].head()


# In[6]:


df = dataset['music']
data = [tuple(x) for x in df.values]
# row is a tuple
def isLocal(row, s):
    a1 = row[0][0]
    n1 = row[0][1]
    a2 = row[1][0]
    n2 = row[1][1]
    return (a1<=a2 and a1+n1>a1+s) or (a1>a2 and a1<a2+n2)


# In[7]:


s_MAX = 2**17
s_MIN = 2**10
s_STEP = 2**11
t_MAX = 100
kst = []
t_ax = [x for x in range(2,t_MAX,8)]
s_ax = [x for x in range(s_MIN, s_MAX, s_STEP)]


# In[ ]:





# In[10]:


for t in range(2,t_MAX,8):
    kst_row = []
    tlist = [x for x in zip(data[t:], data)]
    L = len(tlist)
    for s in range(s_MIN, s_MAX, s_STEP):
        local = [isLocal(x, s) for x in tlist]
        kst_row.append(sum(local) / (L - t))
    print ("Done with t = ", t)
    kst.append(kst_row)


# In[11]:


kst = np.array(kst)


# In[26]:


ax = sns.heatmap(kst, robust=True, linewidth=0, xticklabels=s_ax, yticklabels=t_ax, center=kst.max())
ax.set_ylabel('temporal (instructions)')
ax.set_xlabel('spatial (bytes)')


# In[25]:


plt.close()

