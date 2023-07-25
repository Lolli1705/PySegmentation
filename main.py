from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow,QLabel, 
                             QHBoxLayout, QLayout,QDialog, QGroupBox, QPushButton, QFileDialog, QWidget,
                             QGridLayout, QLineEdit
                             )
import pyqtgraph as pg

from tools.io import get_list_of_files
from tools.processing import array_to_qimage
import os
from skimage import io
from scipy.ndimage import label, generate_binary_structure, convolve
import numpy as np
from segment import Ui_MainWindow
from qimage2ndarray import gray2qimage
import cv2
from tools.functions import conn_segment, save_files
import warnings
import sys

warnings.filterwarnings("ignore")




class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, obj=None, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        

        # Define variables to store data

        self.images_path = [] # path of all the images
        self.folder_name = [] # input folder name
        self.num_files = None # number of file in the input folder
        self.img = None # image to load
        self.slice_num = 0 # number of slice selected
        self.thres = 0 # threshold value 
        self.binary_img = None
        self.kernel = None
        self.kernel_size = 2 
        self.img_erode = None
        self.iterations = 1
        self.connectivity_kernel = None
        self.diagonal = True
        self.conn_threshold = 0
        self.output_path = ""
        
        

        # Loading Box

        self.input_dir_push_button.clicked.connect(self.load_files) # Load button

        # Segmentation
        # Selection
    
        self.slice_spin.valueChanged.connect(self.slice_num_change) # Select slice number
        
        self.threshold_spin.setMinimum(0) # Set the value of threshold to 0 then we set the maximum

        self.load_img.clicked.connect(self.load_im) # To load the image and retrieve the max gray value for the theshold

        # Parameters

        self.erosion_k_spin.valueChanged.connect(self.erosion_kernel)
        self.erosion_k_spin.setMaximum(4)

        self.erosion_it_spin.valueChanged.connect(self.erosion_it)
        self.erosion_it_spin.setMaximum(4)

        self.prew_threshold.clicked.connect((self.show_img_threshold)) # Button to show the image threshold 

        self.prew_erosion.clicked.connect(self.show_img_erosion)
        self.connectivity_threshold_spin.setRange(0,10000000)
        self.connectivity_threshold_spin.valueChanged.connect(self.set_conn_thre)

        self.connectivity_spin.valueChanged.connect(self.set_kernel_size)
        self.prew_connectivity.clicked.connect(self.show_img_erosion_connectivity)

        # Save

        self.select_output_dir.clicked.connect(self.sel_output)
        self.save.clicked.connect(self.save_images)




        #self.prew_erosion.clicked.connect(self.show_img)
        


    #####################
    ######FUNCTIONS######
    #####################
    
    def load_files(self):

        caption = "Select input folder"
        initial_dir = ""
        dialog = QFileDialog()
        dialog.setWindowTitle(caption)
        dialog.setDirectory(initial_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)

        dialog.exec()

        self.images_path = get_list_of_files(dialog.selectedFiles()[0])

        split_dir_path = self.images_path[0].split("/")

        self.folder_name = split_dir_path[-2]

        self.num_files = len(self.images_path)

        self.input_line_edit.setText(dialog.selectedFiles()[0])
        self.folder_edit.setText(self.folder_name)
        self.file_edit.setText(str(self.num_files))       

        if len(self.images_path)>0:

            self.slice_spin.setRange(0, (len(self.images_path)-1))

        

    def slice_num_change(self, val):

        self.slice_num = val



    def load_im(self):

        if isinstance(self.slice_num, int):
            
            self.img = io.imread(self.images_path[self.slice_num])


            if self.img.dtype == 'uint16':

                self.threshold_spin.setMinimum(0)
                self.threshold_spin.setMaximum(65535)
            
            self.thres = self.threshold_spin.valueChanged.connect(self.threshold)

    
    def threshold(self, val_):

        self.thres = val_
        
        


    def show_img_threshold(self):
        self.mask = self.img > self.thres

        self.binary_img = np.ones((self.img.shape[0],self.img.shape[1]), dtype='uint8')*self.mask*255
        self.binary_img = gray2qimage(self.binary_img)
        qim = QImage(self.binary_img)
        self.img_holder.setScaledContents(True)
        self.img_holder.setGeometry(QtCore.QRect(550, 120, int(self.img.shape[1]/1.5), int(self.img.shape[0]/1.5)))
        self.img_holder.setPixmap(QPixmap.fromImage(qim))
        

    def erosion_kernel(self, val_):

        self.kernel_size = val_
    
    def erosion_it(self, val_):

        self.iterations = val_
    
    def show_img_erosion(self):

        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        # Erosion
        self.binary_img = np.ones((self.img.shape[0],self.img.shape[1]), dtype='uint8')*self.mask
        self.img_erode = cv2.erode(self.binary_img, self.kernel, iterations =self.iterations)
        self.img_erode = self.img_erode*255
        self.img_erode = gray2qimage(self.img_erode)
        qim = QImage(self.img_erode)
        self.img_holder.setScaledContents(True)
        self.img_holder.setGeometry(QtCore.QRect(550, 120, int(self.img.shape[1]/1.5), int(self.img.shape[0]/1.5)))
        self.img_holder.setPixmap(QPixmap.fromImage(qim))

    def set_kernel_size(self, val_):

        self.connectivity_kernel = generate_binary_structure(val_,val_)
    
    def set_conn_thre(self, val_):

        self.conn_threshold = val_
    
    def show_img_erosion_connectivity(self):

        self.kernel = np.ones((self.kernel_size, self.kernel_size),np.uint8)

        # Erosion
        self.binary_img = np.ones((self.img.shape[0],self.img.shape[1]))*self.mask
        self.img_erode = cv2.erode(self.binary_img, self.kernel, iterations =self.iterations)
        

        self.img_segm_con = conn_segment(self.img_erode, self.connectivity_kernel, self.conn_threshold)
        

        
        

        self.img_segm_con = gray2qimage(self.img_segm_con, normalize=True)
        qim = QImage(self.img_segm_con)
        # self.img_bin= gray2qimage(self.binary_img, normalize=True)
        # qim = QImage(self.img_bin)
        self.img_holder.setScaledContents(True)
        self.img_holder.setGeometry(QtCore.QRect(550, 120, int(self.img.shape[1]/1.5), int(self.img.shape[0]/1.5)))
        self.img_holder.setPixmap(QPixmap.fromImage(qim))
    
    def sel_output(self):

        caption = "Select input folder"
        initial_dir = ""
        dialog = QFileDialog()
        dialog.setWindowTitle(caption)
        dialog.setDirectory(initial_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)

        dialog.exec()

        self.output_path = dialog.selectedFiles()[0]
        self.output_line_edit.setText(self.output_path)



    def save_images(self):
       
        save_files(output_path=self.output_path,
                    file_names=self.images_path,
                    threshold= self.thres,
                    kernel=self.kernel,
                    iterations=self.iterations,
                    conn_kernel=self.connectivity_kernel,
                    conn_threshold=self.conn_threshold)
            

       
        
app=QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()
