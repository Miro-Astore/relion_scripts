### Use this to perform a 2d class average of a stack of images derived from simulations of single particle cryoEM images.
# It is recomended to run this script in a cluster environment with a large amount of memory. By default this script loads the full stack into memory which is often prohibitive.
# Usage
#python relion_scripts/quick_2d_class_average.py --stack stack_name.mrcs --star template_particles.star -o out_directory
# if -o is not specified then this program will make a direcetory called stack_name_2d_averages as a working directory

import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
from functools import reduce
import matplotlib.gridspec as gridspec
import re
import os 
import pdb
import rlcompleter
import subprocess
import mrcfile

pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

# arguments for directory. 
parser = argparse.ArgumentParser()
parser.add_argument('--stack','-m',dest='stack',type=str)
parser.add_argument('--star-template','-s',dest='star',type=str)
parser.add_argument('-o',dest='out',default=None,type=str)

args = parser.parse_args()

stack_name = args.stack.split('/')[-1]
stack_name = stack_name [:-5]
#setting up run directory, checking files
if args.out == None:
    out_dir_name = stack_name + '_2d_averages' 
    os.makedirs(stack_name + '_2d_averages' , exist_ok=True)
    out_dir = os.path.abspath(out_dir_name)
else:
    os.mkdir(args.out)
    out_dir = os.path.abspath(args.out)

stack_location = os.path.abspath (args.stack)
star_location = os.path.abspath (args.star)

if os.path.exists(stack_location) == False:
    raise ValueError ('Stack file doesn\'t exist, check your path.')

if os.path.exists(star_location) == False:
    raise ValueError ('Star file doesn\'t exist, check your path.')

#writing star file for this specfic stack.
new_star_filename  = out_dir + '/' + stack_name + '.star'
new_star_file = open(new_star_filename, 'w+')
star_file = open(star_location,'r')
Lines = star_file.readlines() 
i = 0 
for line in Lines : 
    if re.search ('@.*?\s', line) != None: 
        i = i + 1
        num_string = "{:06d}".format(i)
        #line = re.sub ('\D@.*?\s', num_string + '@' + str(stack_location) + ' ' , line) 
        line = re.sub ('[0-9]*@.*?\s', num_string + '@' + str(stack_location) + ' ' , line) 
        print(line)
    new_star_file.write(line)

os.makedirs( str(out_dir) +  '/2d_class/',exist_ok = True)

#relion_2d_command = 'relion_refine --o ' + str(out_dir) +  '  --grad --class_inactivity_threshold 0.1 --grad_write_iter 10 --iter 10 --i ' + new_star_filename +  '  --dont_combine_weights_via_disc --preread_images  --pool 3 --pad 2  --ctf  --tau2_fudge 2 --particle_diameter 300 --K 20 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu "0"  '

relion_2d_command = 'relion_refine --o ' + str(out_dir) +  '/2d_class/  --grad --class_inactivity_threshold 0.1 --grad_write_iter 1 --iter 20 --i ' + new_star_filename +  '  --dont_combine_weights_via_disc   --pool 3 --pad 2  --ctf  --tau2_fudge 2 --particle_diameter 300 --K 20 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu'

subprocess.run(relion_2d_command, shell=True, check=True) 
file_list = (os.listdir(out_dir))
mrcs_files = []

for i in file_list:
    if 'classes.mrcs'  in i:
        mrcs_files.append(i)
last_mrcs_file  = out_dir + '/' + str((np.sort(mrcs_files))[-1])

class_data = mrcfile.open(last_mrcs_file)

num_classes = len(class_data.data)

def factors(n):
        return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def choose_most_square (n) :
    target = np.sqrt(n)

facts = np.sort(list(factors(num_classes)))

#determining rows and cols arrangement, more annoying than you'd think
num_plots_rows=0
num_plots_cols=0

#if a square number we'll have an odd number of factors so set number of rows and cols to the same thing
if (len(facts) % 2) == 0:
    num_plots_rows=facts[int(len(facts)/2)]
    num_plots_cols=facts[int(len(facts)/2)-1]
else:
    num_plots_rows=facts[int((len(facts)-1)/2)]
    num_plots_cols=facts[int((len(facts)-1)/2)]

row_arr=range(num_plots_rows)
col_arr=range(num_plots_cols)

row_arr_t=list(itertools.chain.from_iterable(itertools.repeat(x, num_plots_cols) for x in row_arr))
row_arr=row_arr_t

col_arr=list(col_arr)*num_plots_rows

AX=gridspec.GridSpec(num_plots_rows,num_plots_cols)

for j in range(num_classes):
    row_place=row_arr[j]
    col_place=col_arr[j]
    plt.subplot2grid((num_plots_rows,num_plots_cols),(row_place,col_place),colspan=1,rowspan=1)
    plt.imshow(class_data.data[j],cmap='gray')

plt.tight_layout()
plt.savefig(out_dir + '/' + 'classes_viz.pdf')
#plt.show()
