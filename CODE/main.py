import os
import sys

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from tensorflow.keras.applications import DenseNet201, ResNet50, VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DISEASE_CLASSES = {
    0:'The Patient is suspected to have COVID symptoms',
    1:  'The Patient does not have COVID',
}

# //class
class Covid_19_Detection(QMainWindow):
    def __init__(self):
        super(Covid_19_Detection, self).__init__()
        loadUi('MainWindow.ui', self)
        os.system('cls')

        self.train_algo_comboBox.activated.connect(self.Show_training_results)
        self.browse_pushButton.clicked.connect(self.BrowseFileDialog)
        self.Prediction_pushButton.clicked.connect(self.Classification_Function)

        self.qm = QMessageBox()

    @pyqtSlot()
    # for showing results
    def Show_training_results(self):
        self.train_algo = str(self.train_algo_comboBox.currentText())

        if self.train_algo == 'DenseNet':
            img_1 = cv2.imread('./training/densenet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./training/densenet/conf_mat.png')
            self.DisplayImage(img_2, 2)
            text = open('./training/densenet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)


        elif self.train_algo == 'ResNet':
            img_1 = cv2.imread('./training/resnet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./training/resnet/conf_mat.png')
            self.DisplayImage(img_2, 2)
            text = open('./training/resnet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)


        elif self.train_algo == 'VGG':
            img_1 = cv2.imread('./training/vgg/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./training/vgg/conf_mat.png')
            self.DisplayImage(img_2, 2)
            text = open('./training/vgg/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)

    @pyqtSlot()
    def BrowseFileDialog(self):
        self.fname, filter = QFileDialog.getOpenFileName(self, 'Open image File', '.\\', "image Files (*.*)")
        if self.fname:
            self.LoadImageFunction(self.fname)
        else:
            print("No Valid File selected.")

    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 0)

    def DisplayImage(self, img, window):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 0:
            self.query_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.query_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.query_imglabel.setScaledContents(True)
        elif window == 1:
            self.model_eval_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.model_eval_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.model_eval_imglabel.setScaledContents(False)
        elif window == 2:
            self.confusion_matrix_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.confusion_matrix_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.confusion_matrix_imglabel.setScaledContents(False)
        elif window == 3:
            self.fraction_incorrect_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.fraction_incorrect_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.fraction_incorrect_imglabel.setScaledContents(False)

    @pyqtSlot()
    def Classification_Function(self):

        self.prediction_algo = str(self.prediction_algo_comboBox.currentText())

        if self.prediction_algo == 'DenseNet':
            self.DenseNet_Prediction()
        elif self.prediction_algo == 'ResNet':
            self.ResNet_Prediction()
        elif self.prediction_algo == 'VGG':
            self.VGG_Prediction()
        else:
            ret = self.qm.information(self, 'Error !', 'No Algo Selected !\nPlease Select an algorithm', self.qm.Close)

    # ----------------------------------------------------------------------------------------------------------------------

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    # ----------------------------------------------------------------------------------------------------------------------

    def Load_Training_Model(self, model_name):

        INIT_LR = 1e-3
        EPOCHS = 30
        BS = 8

        if model_name == "resnet":
            baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        elif model_name == "vgg":
            baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        elif model_name == "densenet":
            baseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        if model_name == "resnet":
            model.load_weights('./training/resnet/Resnet50_covid_model.h5')

        elif model_name == "vgg":
            model.load_weights('./training/vgg/VGG_covid_model.h5')

        elif model_name == "densenet":
            model.load_weights('./training/densenet/DenseNet201_covid_model.h5')

        # model.summary()

        return model

    # ----------------------------------------------------------------------------------------------------------------------

    def Predict_Test_Image_File(self, model, model_name):

        print(self.fname)

        img_width, img_height = 224, 224
        img = image.load_img(self.fname, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        img = np.expand_dims(x, axis=0)

        pred = model.predict(img)
        print(pred)

        index = np.argmax(pred, axis=1)[0]

        disease_name = DISEASE_CLASSES[index]
        print(disease_name)

        self.prediction_result_label.setText(disease_name)

    # ----------------------------------------------------------------------

    def DenseNet_Prediction(self):
        model = self.Load_Training_Model('densenet')
        self.Predict_Test_Image_File(model, 'densenet')

    # ----------------------------------------------------------------------

    def ResNet_Prediction(self):
        model = self.Load_Training_Model('resnet')
        self.Predict_Test_Image_File(model, 'resnet')

    # ----------------------------------------------------------------------

    def VGG_Prediction(self):
        model = self.Load_Training_Model('vgg')
        self.Predict_Test_Image_File(model, 'vgg')


# ----------------------------------------------------------------------


''' ------------------------ MAIN Function ------------------------- '''

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Covid_19_Detection()
    window.show()
    sys.exit(app.exec_())
