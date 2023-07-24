# Form implementation generated from reading ui file 'seg09.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1644, 805)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolBox = QtWidgets.QToolBox(parent=self.centralwidget)
        self.toolBox.setGeometry(QtCore.QRect(10, 0, 521, 471))
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 521, 403))
        self.page.setObjectName("page")
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.page)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 0, 471, 171))
        self.groupBox_4.setObjectName("groupBox_4")
        self.input_dir_label = QtWidgets.QLabel(parent=self.groupBox_4)
        self.input_dir_label.setGeometry(QtCore.QRect(10, 33, 70, 16))
        self.input_dir_label.setObjectName("input_dir_label")
        self.input_line_edit = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.input_line_edit.setGeometry(QtCore.QRect(70, 29, 281, 21))
        self.input_line_edit.setObjectName("input_line_edit")
        self.folder_label = QtWidgets.QLabel(parent=self.groupBox_4)
        self.folder_label.setGeometry(QtCore.QRect(10, 63, 81, 20))
        self.folder_label.setObjectName("folder_label")
        self.input_dir_push_button = QtWidgets.QPushButton(parent=self.groupBox_4)
        self.input_dir_push_button.setGeometry(QtCore.QRect(350, 25, 113, 32))
        self.input_dir_push_button.setObjectName("input_dir_push_button")
        self.file_edit = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.file_edit.setGeometry(QtCore.QRect(120, 90, 51, 21))
        self.file_edit.setObjectName("file_edit")
        self.file_number = QtWidgets.QLabel(parent=self.groupBox_4)
        self.file_number.setGeometry(QtCore.QRect(10, 93, 101, 16))
        self.file_number.setObjectName("file_number")
        self.folder_edit = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.folder_edit.setGeometry(QtCore.QRect(100, 61, 151, 21))
        self.folder_edit.setObjectName("folder_edit")
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setEnabled(True)
        self.page_2.setGeometry(QtCore.QRect(0, 0, 521, 403))
        self.page_2.setObjectName("page_2")
        self.groupBox = QtWidgets.QGroupBox(parent=self.page_2)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 481, 81))
        self.groupBox.setObjectName("groupBox")
        self.slice_label = QtWidgets.QLabel(parent=self.groupBox)
        self.slice_label.setGeometry(QtCore.QRect(10, 30, 41, 16))
        self.slice_label.setObjectName("slice_label")
        self.slice_spin = QtWidgets.QSpinBox(parent=self.groupBox)
        self.slice_spin.setGeometry(QtCore.QRect(50, 26, 61, 24))
        self.slice_spin.setObjectName("slice_spin")
        self.spinBox_3 = QtWidgets.QSpinBox(parent=self.groupBox)
        self.spinBox_3.setGeometry(QtCore.QRect(80, 80, 61, 24))
        self.spinBox_3.setObjectName("spinBox_3")
        self.load_img = QtWidgets.QPushButton(parent=self.groupBox)
        self.load_img.setGeometry(QtCore.QRect(350, 25, 113, 32))
        self.load_img.setObjectName("load_img")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.page_2)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 90, 481, 171))
        self.groupBox_2.setObjectName("groupBox_2")
        self.threshold_seg = QtWidgets.QLabel(parent=self.groupBox_2)
        self.threshold_seg.setGeometry(QtCore.QRect(10, 30, 71, 16))
        self.threshold_seg.setObjectName("threshold_seg")
        self.connectivity_k_lable = QtWidgets.QLabel(parent=self.groupBox_2)
        self.connectivity_k_lable.setGeometry(QtCore.QRect(10, 90, 141, 16))
        self.connectivity_k_lable.setObjectName("connectivity_k_lable")
        self.connectivity_check = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.connectivity_check.setGeometry(QtCore.QRect(363, 88, 87, 24))
        self.connectivity_check.setObjectName("connectivity_check")
        self.connectivity_threshold = QtWidgets.QLabel(parent=self.groupBox_2)
        self.connectivity_threshold.setGeometry(QtCore.QRect(10, 120, 141, 16))
        self.connectivity_threshold.setObjectName("connectivity_threshold")
        self.erosion_label = QtWidgets.QLabel(parent=self.groupBox_2)
        self.erosion_label.setGeometry(QtCore.QRect(10, 60, 91, 16))
        self.erosion_label.setObjectName("erosion_label")
        self.erosion_label_iter = QtWidgets.QLabel(parent=self.groupBox_2)
        self.erosion_label_iter.setGeometry(QtCore.QRect(170, 58, 61, 20))
        self.erosion_label_iter.setObjectName("erosion_label_iter")
        self.prew_erosion = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.prew_erosion.setGeometry(QtCore.QRect(350, 55, 113, 32))
        self.prew_erosion.setObjectName("prew_erosion")
        self.threshold_spin = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.threshold_spin.setGeometry(QtCore.QRect(90, 24, 91, 24))
        self.threshold_spin.setObjectName("threshold_spin")
        self.erosion_k_spin = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.erosion_k_spin.setGeometry(QtCore.QRect(107, 56, 51, 24))
        self.erosion_k_spin.setObjectName("erosion_k_spin")
        self.erosion_it_spin = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.erosion_it_spin.setGeometry(QtCore.QRect(240, 56, 51, 24))
        self.erosion_it_spin.setObjectName("erosion_it_spin")
        self.connectivity_spin = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.connectivity_spin.setGeometry(QtCore.QRect(140, 86, 51, 21))
        self.connectivity_spin.setObjectName("connectivity_spin")
        self.prew_threshold = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.prew_threshold.setGeometry(QtCore.QRect(350, 25, 113, 32))
        self.prew_threshold.setObjectName("prew_threshold")
        self.connectivity_threshold_spin = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.connectivity_threshold_spin.setGeometry(QtCore.QRect(160, 116, 51, 24))
        self.connectivity_threshold_spin.setMaximum(1000000000)
        self.connectivity_threshold_spin.setObjectName("connectivity_threshold_spin")
        self.prew_connectivity = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.prew_connectivity.setGeometry(QtCore.QRect(350, 114, 113, 32))
        self.prew_connectivity.setObjectName("prew_connectivity")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.page_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 270, 481, 111))
        self.groupBox_3.setObjectName("groupBox_3")
        self.output_dir_label = QtWidgets.QLabel(parent=self.groupBox_3)
        self.output_dir_label.setGeometry(QtCore.QRect(10, 30, 91, 16))
        self.output_dir_label.setObjectName("output_dir_label")
        self.output_line_edit = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.output_line_edit.setGeometry(QtCore.QRect(80, 28, 271, 21))
        self.output_line_edit.setObjectName("output_line_edit")
        self.select_output_dir = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.select_output_dir.setGeometry(QtCore.QRect(350, 23, 113, 32))
        self.select_output_dir.setObjectName("select_output_dir")
        self.save = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.save.setGeometry(QtCore.QRect(350, 53, 113, 32))
        self.save.setObjectName("save")
        self.toolBox.addItem(self.page_2, "")
        self.img_holder = QtWidgets.QLabel(parent=self.centralwidget)
        self.img_holder.setGeometry(QtCore.QRect(550, 10, 611, 531))
        self.img_holder.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.img_holder.setText("")
        self.img_holder.setObjectName("img_holder")
        self.connectivitylist = QtWidgets.QListView(parent=self.centralwidget)
        self.connectivitylist.setGeometry(QtCore.QRect(30, 490, 451, 192))
        self.connectivitylist.setObjectName("connectivitylist")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1644, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PySegmentation"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Loading"))
        self.input_dir_label.setText(_translate("MainWindow", "Input dir:"))
        self.folder_label.setText(_translate("MainWindow", "Folder name:"))
        self.input_dir_push_button.setText(_translate("MainWindow", "Select folder"))
        self.file_number.setText(_translate("MainWindow", "Number of files:"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "Input"))
        self.groupBox.setTitle(_translate("MainWindow", "Selection"))
        self.slice_label.setText(_translate("MainWindow", "Slice:"))
        self.load_img.setText(_translate("MainWindow", "Load image"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Parameters"))
        self.threshold_seg.setText(_translate("MainWindow", "Threshold:"))
        self.connectivity_k_lable.setText(_translate("MainWindow", "Connectivity Kernel"))
        self.connectivity_check.setText(_translate("MainWindow", "Diagonal"))
        self.connectivity_threshold.setText(_translate("MainWindow", "Connectivity Threshold"))
        self.erosion_label.setText(_translate("MainWindow", "Erosion Kernel:"))
        self.erosion_label_iter.setText(_translate("MainWindow", "Iterations:"))
        self.prew_erosion.setText(_translate("MainWindow", "Preview"))
        self.prew_threshold.setText(_translate("MainWindow", "Preview"))
        self.prew_connectivity.setText(_translate("MainWindow", "Preview"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Exporting and Saving"))
        self.output_dir_label.setText(_translate("MainWindow", "Output dir:"))
        self.select_output_dir.setText(_translate("MainWindow", "Select folder"))
        self.save.setText(_translate("MainWindow", "Save"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Segmentation"))
