# Firstly ensure the execution path is the same as the file path
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pyplot as plt

# for converting pdfs to black and white
from os.path import join
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path # https://pypi.org/project/pdf2image/
from img2pdf import convert # https://pypi.org/project/img2pdf/
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random

# Also plot a BW figure for a preview of the colour PDF printed in BW if True
PLOT_BW = True

# Output various details of plotting if True
VERBOSE = False

# Output a message when a plot has been completed if True
VERBOSE_PLOTTED = True

STANDARD_ONE_FIG_WIDTH_CM = 9/2.54
STANDARD_TWO_FIG_WIDTH_CM = 16/2.54
STANDARD_THREE_FIG_WIDTH_CM = 23/2.54

blue = "steelblue"
red = "firebrick"
green = "green"
cyan = "cyan"
black = "black"
grey = "grey"
lightgrey = "lightgrey"
magenta = "magenta"

heat_plot_path_col = cyan
line_plot_cols = [blue, red, black, green, cyan, magenta]
scatter_cols = [blue, red]
histogram_cols = [lightgrey]

def get_heat_color_map(no_bins = 20):
    # heat plots will be coarser with less bins
    colors = ['gold', 'red', 'darkred'] # colours used in the heat plots 
    return LinearSegmentedColormap.from_list('my_list', colors, N = no_bins)

# Common function for creating plots of a certain size with appropriate dpi setting
def plot_figure(nrow = 1, ncol = 1, width_scale = 1, height_scale = 1, book_scale = 1, pad = 3):
    """    
    Args:    
    nrow (int): The number of plot rows.
    ncol (int): The number of plot columns.
    """
    
    if ncol == 1:
        width = STANDARD_ONE_FIG_WIDTH_CM             
    elif ncol == 2:
        width = STANDARD_TWO_FIG_WIDTH_CM             
    else:
        width = STANDARD_THREE_FIG_WIDTH_CM       
       
    individual_fig_width = width / ncol
    height = individual_fig_width * nrow
    
    width *= width_scale
    height *= height_scale
      
    # Adjust font size, so all will be the same once scaled again according to scaling for the book
    initial_font_size = 8 
    rescaled_font_size = (initial_font_size * width)/(book_scale * STANDARD_ONE_FIG_WIDTH_CM)
    set_all_font_sizes(rescaled_font_size)
    
    plt.rcParams['font.size'] = initial_font_size
    plt.rcParams['font.sans-serif'] = "Times New Roman" 
    plt.rcParams['font.family'] = "sans-serif"   
    plt.rcParams['mathtext.fontset'] = "cm" # make math symbols simliar to LaTeX, curly x's etc.
    
    # Lengths are converted to cm here   
    fig, ax = plt.subplots(nrow, ncol, figsize=(width, height), dpi=1200)
    
    # Add padding for multiple plots
    plt.tight_layout(pad = pad, w_pad = 3.5, h_pad = 4)
    
    return fig, ax

def set_all_font_sizes(font_size):
    SMALL_SIZE = font_size * 0.8
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size * 1.2

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
   
     
# Common function for saving plots
def save_figure(filename):
    """    
    Args:    
    filename (string): The name of the file.   
    """
   
    # Create directory if it does not exisit already
    if not os.path.exists("Figures"):  
        os.makedirs("Figures") 
    
    # Save all images in the Images folder
    plt.savefig(join('Figures', filename)) 
     
    # Close the plot
    plt.close()
    
    if VERBOSE_PLOTTED:
        print("Figure plotted: " + filename + " ")
        
    if PLOT_BW:
        save_figure_BW(filename)
         
# Plot figure in greyscale (black and white)
def save_figure_BW(filename):            
    with TemporaryDirectory() as temp_dir: # Saves images temporarily in disk rather than RAM to speed up parsing
        
        # Load pdf file
        images = convert_from_path(
            join('Figures', filename),
            output_folder=temp_dir,
            grayscale=True,
            fmt="jpeg",
            thread_count=4,
            dpi=1200
        )
       
        # convert image to a BW png image file
        image_list = list()
        for page_number in range(1, len(images) + 1):
            path =  temp_dir + 'page_' + str(page_number) + '.jpeg'
            image_list.append(path)
            images[page_number-1].save(path, 'JPEG') 

        # Create directory if it does not exisit already
        if not os.path.exists('Figures-BW-low-quality'):  
            os.makedirs('Figures-BW-low-quality') 
            
        # Convert BW image to a pdf file
        with open(join('Figures-BW-low-quality', 'BW-' + filename), 'bw') as gray_pdf:
            gray_pdf.write(convert(image_list))

        if VERBOSE_PLOTTED:
            print("\tFigure plotted: BW-" + filename + " ")

#Set random seed for both numpy.random and random packages 
def seed(value):
    np.random.seed(value)
    random.seed(value)