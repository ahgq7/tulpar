"""
@author: ahgq7
"""

import io
import shutil
import tkinter as tk
from random import randrange
from tkinter import ttk
import os
from os import listdir
from os.path import isfile, join
from tkinter import filedialog
from tkinter import messagebox
from keras.applications.densenet import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K, models
import sys
import threading
from threading import Thread

window = tk.Tk()
window.geometry("1080x720")
window.wm_title("Tulpar - Model Eğitici")
window.resizable(False, False)
window.configure(bg="snow")
s = Style()
s.configure("My.TFrame", background="snow")
s2 = Style()
s2.configure("TNotebook", background="white smoke")

window.sourceFolder = ''
window.sourceFile = ''
window.modelPath = ''
window.photoPath = ''
window.nb_train_samples = 0
window.nb_validation_samples = 0
old_stdout = sys.stdout
window.new_stdout = io.StringIO()
sys.stdout = window.new_stdout
window.output3 = ""
window.epoch = 50
window.batch = 16
window.img_width = 150
window.img_height = 150
window.activation = "relu"
window.output = {}
window.resimSayisi = 0
window.modelNo = 1
window.kill = 0
window.modelPath = ""
window.modelSayisi = 0
window.predictions = {}

#window.bind('<Escape>', lambda e: cikis())


def foo():
    t = threading.Timer(0.01, foo)
    if window.kill == 0:
        if window.output == window.new_stdout.getvalue():
            pass
        else:
            window.output = window.new_stdout.getvalue()
            output2 = window.output.splitlines()
            ciktiText.delete('1.0', END)
            ciktiText.insert(1.0, "%s\n" % window.output3)
            output3 = str(output2[-1:])
            try:
                if output3[2] == 'E':
                    window.output3 = output3
            except:
                pass
            ciktiText.insert("end", output2[-1:])
        t.start()
    else:
        exit(1)


def modelEgit():
    process = Thread(target=modelEgit2, args=())
    process.start()


def modelEgit2():
    sys.stdout = window.new_stdout

    img_width, img_height = window.img_width, window.img_height
    train_data_dir = "%s/train" % window.sourceFolder
    validation_data_dir = "%s/test" % window.sourceFolder
    nb_train_samples = window.nb_train_samples
    # nb_validation_samples = window.nb_validation_samples
    epochs = window.epoch
    batch_size = window.batch

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=True,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation=window.activation, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=window.activation))
    model.add(MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=window.activation))

    model.add(Flatten())
    model.add(Dense(64, activation=window.activation))
    model.add(Dense(10))

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_generator, steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs, validation_data=validation_generator)

    try:
        model.save("%s/model" % window.sourceFolder + "%i-" % window.modelNo + "%s.h5" % window.activation)
        treeview2.insert("in", window.modelNo * 15,
                         "%s/model" % window.sourceFolder + "%i-" % window.modelNo + "%s.h5" % window.activation,
                         text="model%i-" % window.modelNo + "%s" % window.activation)

        window.modelPath = "%s/model" % window.sourceFolder + "%i-" % window.modelNo + "%s.h5" % window.activation
        window.modelSayisi = window.modelSayisi + 1
    except:
        randomNuber = randrange(0, 9999999, 1)
        model.save("%s/model" % window.sourceFolder + "%i-" % randomNuber + "%s.h5" % window.activation)
        treeview2.insert("in", window.modelNo * 15,
                         "%s/model" % window.sourceFolder + "%i-" % randomNuber + "%s.h5" % window.activation,
                         text="model%i-" % randomNuber + "%s" % window.activation)

        window.modelPath = "%s/model" % window.sourceFolder + "%i-" % randomNuber + "%s.h5" % window.activation
        window.modelSayisi = window.modelSayisi + 1

    window.modelNo = window.modelNo + 1

    acc = np.array(history.history["accuracy"])
    val_acc = np.array(history.history["val_accuracy"])

    loss = np.array(history.history["loss"])
    val_loss = np.array(history.history["val_loss"])

    epochs_range = np.array(range(epochs))

    for widget in gorselData.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(8, 5), dpi=80)

    plot1 = fig.add_subplot(121)
    plot1.plot(epochs_range, acc, label="Train Accuracy")
    plot1.plot(epochs_range, val_acc, label="Test Accuracy")
    plot1.legend(loc="upper right")
    # plot1.title("Train ve Test Accuracy")

    plot1 = fig.add_subplot(122)
    plot1.plot(epochs_range, loss, label="Train Loss")
    plot1.plot(epochs_range, val_loss, label="Test Loss")
    plot1.legend(loc="upper right")
    # plot1.title("Train ve Test Loss")

    canvas = FigureCanvasTkAgg(fig, master=gorselData)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, gorselData)
    toolbar.update()
    canvas.get_tk_widget().pack()
    del model


def navigationOlustur():
    x = treeview.get_children()
    for item in x:
        treeview.delete(item)

    treeview.insert("", 0, "test", text="test")
    treeview.insert("", 1, "train", text="train")

    i = 2

    trainpath = "%s/train/" % window.sourceFolder
    testpath = "%s/test/" % window.sourceFolder

    traindirs = [f for f in listdir(trainpath) if
                 not isfile(join(trainpath, f))]

    testdirs = [f for f in listdir(testpath) if
                not isfile(join(testpath, f))]

    window.predictions = testdirs
    window.predictions.sort()

    window.nb_validation_samples = 0
    window.nb_train_samples = 0

    for each in testdirs:
        treeview.insert("test", i, each, text=each)
        testfiles = [f for f in listdir(testpath + each) if
                     isfile(join(testpath + each, f))]
        for each2 in testfiles:
            window.nb_validation_samples = window.nb_validation_samples + 1
            treeview.insert(each, i, testpath + each + "/" + each2, text=each2)
            i = i + 1
    i = i + 1

    for each in traindirs:
        treeview.insert("train", i, each + "1", text=each)
        trainfiles = [f for f in listdir(trainpath + each) if
                      isfile(join(trainpath + each, f))]
        for each2 in trainfiles:
            window.nb_train_samples = window.nb_train_samples + 1
            treeview.insert(each + "1", i, trainpath + each + "/" + each2, text=each2)
            i = i + 1
    i = i + 1
    print("%i" % window.nb_train_samples + " %i" % window.nb_validation_samples)


def chooseDir():
    window.sourceFolder = filedialog.askdirectory(parent=window, initialdir="", title='Klasör Konumu Seçin')
    navigationOlustur()


def navigationOlustur2():
    window.resimSayisi = window.resimSayisi + 1
    treeview.insert("", window.resimSayisi * 9528, window.sourceFile, text="Resim %i" % window.resimSayisi)


def chooseFile():
    window.sourceFile = filedialog.askopenfilename(parent=window, initialdir="", title='Dosya Seçin')
    navigationOlustur2()


def cikis():
    window.kill = 1
    window.destroy()
    exit(1)


window.protocol("WM_DELETE_WINDOW", cikis)


def imageResize(img):
    basewidth = 350
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def resimAc(event):
    item = treeview.identify("item", event.x, event.y)
    for widget in gorsel.winfo_children():
        widget.destroy()
    window.photoPath = item
    img = Image.open(window.photoPath)
    img = imageResize(img)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(gorsel, image=img)
    panel.image = img
    panel.pack()


menubar = tk.Menu(window)
window.config(menu=menubar)

file = tk.Menu(menubar)
info = tk.Menu(menubar)

menubar.add_cascade(label="Dosya", menu=file)
menubar.add_cascade(label="Hakkında", menu=info)


def authorWho():
    messagebox._show(title="Hazırlayan", message="Bu yazılım, ahgq7 tarafından açık kaynak olarak yazılmıştır.")


info.add_command(label="Hazırlayan", command=authorWho)

files = ttk.Notebook(window, width=240, height=500, style="TNotebook")
files.place(x=0, y=0)

shows = ttk.Notebook(window, width=600, height=500, style="TNotebook")
shows.place(x=240, y=0)

opts = ttk.Frame(window, width=240, height=500, style="My.TFrame", relief=tk.SUNKEN)
opts.place(x=840, y=0)

codes = ttk.Notebook(window, width=1080, height=250, style="TNotebook")
codes.place(x=0, y=440)

dosya = ttk.Frame(files, style="My.TFrame")

treeview = ttk.Treeview(dosya, height=19)
treeview.columnconfigure(2, weight=40)
treeview.place(x=17, y=2)
treeview.bind("<Double-1>", resimAc)


def modelSec(event):
    item = treeview2.identify("item", event.x, event.y)
    window.modelPath = item


def modelAktar():
    modelYukle = filedialog.askopenfilename(parent=window, initialdir="/", title='Dosya Seçin')
    treeview2.insert("out", window.modelSayisi, modelYukle, text="Model-%i" % window.modelSayisi)


def testTrainOlustur():
    window.sourceFolder = filedialog.askdirectory(parent=window, initialdir="", title='Klasör Konumu Seçin')

    dirs = [f for f in listdir(window.sourceFolder) if
            not isfile(join(window.sourceFolder, f))]

    if not os.path.exists(window.sourceFolder + "/train"):
        os.makedirs(window.sourceFolder + "/train")
    if not os.path.exists(window.sourceFolder + "/test"):
        os.makedirs(window.sourceFolder + "/test")

    for each in dirs:
        if not os.path.exists(window.sourceFolder + "/train/%s" % each):
            os.makedirs(window.sourceFolder + "/train/%s" % each)
        if not os.path.exists(window.sourceFolder + "/test/%s" % each):
            os.makedirs(window.sourceFolder + "/test/%s" % each)

    for each in dirs:
        files = [f for f in listdir(window.sourceFolder + "/" + each) if
                 isfile(join(window.sourceFolder + "/" + each, f))]
        k = len(files)
        k1 = k
        for each2 in files:
            if (k1 * 33 / 100) < k:
                shutil.move(window.sourceFolder + "/" + each + "/" + each2, window.sourceFolder + "/train/%s" % each)
            else:
                shutil.move(window.sourceFolder + "/" + each + "/" + each2, window.sourceFolder + "/test/%s" % each)
            k = k - 1
        shutil.rmtree(window.sourceFolder + "/" + each)

    navigationOlustur()


model = ttk.Frame(files, style="My.TFrame")

treeview2 = ttk.Treeview(model, height=19)
treeview2.columnconfigure(2, weight=40)
treeview2.place(x=17, y=2)
treeview2.bind("<Double-1>", modelSec)
treeview2.insert("", 0, "in", text="Programda Eğitilen")
treeview2.insert("", 1, "out", text="Dışardan Alınan")

file.add_command(label="Train Klasörü Seç", command=chooseDir)
file.add_command(label="Test İçin Fotoğraf Seç", command=chooseFile)
file.add_command(label="Test-Train Oluştur", command=testTrainOlustur)
file.add_command(label="Modeli İçe Aktar", command=modelAktar)
file.add_command(label="Çıkış", command=cikis)

files.add(dosya, text="Dosyalar")
files.add(model, text="Modeller")

konsol = ttk.Frame(codes, style="My.TFrame")
cikti = ttk.Frame(codes, style="My.TFrame")

codes.add(cikti, text="Output")
codes.add(konsol, text="Konsol")

ciktiText = tk.Text(cikti, wrap=tk.WORD, state="normal")
ciktiText.pack(fill="both")


def chngEpoch(self):
    window.epoch = int(konsolEntryEpoch.get())
    print("Epoch değeri güncellendi: %i" % window.epoch)
    lEpoch.config(text="Epochs: %i" % window.epoch)


konsolEntryEpochLabel = tk.Label(konsol, text="Epochs: ").place(x=5, y=10)
konsolEntryEpoch = tk.Entry(konsol, width=10)
konsolEntryEpoch.place(x=70, y=10)
konsolEntryEpoch.bind("<Return>", chngEpoch)


def chngBatch(self):
    window.batch = int(konsolEntryBatch.get())
    print("Batch değeri güncellendi: %i" % window.batch)
    lBatch.config(text="Batch Size: %i" % window.batch)


konsolEntryBatchLabel = tk.Label(konsol, text="Batch: ").place(x=5, y=40)
konsolEntryBatch = tk.Entry(konsol, width=10)
konsolEntryBatch.place(x=70, y=40)
konsolEntryBatch.bind("<Return>", chngBatch)


def chngWidth(self):
    window.img_width = int(konsolEntryWidth.get())
    print("Width değeri güncellendi: %i" % window.img_width)
    lImgWidth.config(text="Img Width: %i" % window.img_width)


konsolEntryWidthLabel = tk.Label(konsol, text="Width: ").place(x=5, y=70)
konsolEntryWidth = tk.Entry(konsol, width=10)
konsolEntryWidth.place(x=70, y=70)
konsolEntryWidth.bind("<Return>", chngWidth)


def chngHeight(self):
    window.img_height = int(konsolEntryHeight.get())
    print("Height değeri güncellendi: %i" % window.img_height)
    lImgHeight.config(text="Img Height: %i" % window.img_height)


konsolEntryHeightLabel = tk.Label(konsol, text="Height: ").place(x=5, y=100)
konsolEntryHeight = tk.Entry(konsol, width=10)
konsolEntryHeight.place(x=70, y=100)
konsolEntryHeight.bind("<Return>", chngHeight)

method = tk.StringVar()

konsolReluRadio = tk.Radiobutton(konsol, text="relu", value="relu", variable=method)
konsolReluRadio.place(x=5, y=130)
konsolSigmoidRadio = tk.Radiobutton(konsol, text="sigmoid", value="sigmoid", variable=method)
konsolSigmoidRadio.place(x=5, y=190)
konsolTanhRadio = tk.Radiobutton(konsol, text="tanh", value="tanh", variable=method)
konsolTanhRadio.place(x=105, y=130)
konsolSeluRadio = tk.Radiobutton(konsol, text="selu", value="selu", variable=method)
konsolSeluRadio.place(x=5, y=160)
konsolEluRadio = tk.Radiobutton(konsol, text="elu", value="elu", variable=method)
konsolEluRadio.place(x=105, y=160)

konsolSoftmaxRadio = tk.Radiobutton(konsol, text="softmax", value="softmax", variable=method)
konsolSoftmaxRadio.place(x=205, y=130)
konsolSoftplusRadio = tk.Radiobutton(konsol, text="softplus", value="softplus", variable=method)
konsolSoftplusRadio.place(x=205, y=160)
konsolSoftsignRadio = tk.Radiobutton(konsol, text="softsign", value="softsign", variable=method)
konsolSoftsignRadio.place(x=105, y=190)
konsolExponentialRadio = tk.Radiobutton(konsol, text="exponential", value="exponential", variable=method)
konsolExponentialRadio.place(x=205, y=190)


def chngActivation(self):
    window.activation = method.get()
    print("Aktivasyon fonksiyonu güncellendi: %s" % window.activation)
    lActivation.config(text="Activation: %s" % window.activation)


konsolReluRadio.bind("<Button-1>", chngActivation)
konsolSigmoidRadio.bind("<Button-1>", chngActivation)
konsolTanhRadio.bind("<Button-1>", chngActivation)
konsolSeluRadio.bind("<Button-1>", chngActivation)
konsolEluRadio.bind("<Button-1>", chngActivation)
konsolSoftmaxRadio.bind("<Button-1>", chngActivation)
konsolSoftplusRadio.bind("<Button-1>", chngActivation)
konsolSoftsignRadio.bind("<Button-1>", chngActivation)
konsolExponentialRadio.bind("<Button-1>", chngActivation)

gorsel = ttk.Frame(shows, style="My.TFrame")
gorselData = ttk.Frame(shows, style="My.TFrame")
webcam = ttk.Frame(shows, style="My.TFrame")

shows.add(gorsel, text="Görsel")
shows.add(gorselData, text="Eğitim Grafiği")
shows.add(webcam, text="WebCam")

lEpoch = tk.Label(opts, text="Epochs: %i" % window.epoch)
lEpoch.place(x=15, y=22)
lBatch = tk.Label(opts, text="Batch Size: %i" % window.batch)
lBatch.place(x=15, y=42)
lImgWidth = tk.Label(opts, text="Img Width: %i" % window.img_width)
lImgWidth.place(x=15, y=62)
lImgHeight = tk.Label(opts, text="Img Height: %i" % window.img_height)
lImgHeight.place(x=15, y=82)
lActivation = tk.Label(opts, text="Activation: %s" % window.activation)
lActivation.place(x=15, y=102)

bEgit = tk.Button(opts, text="Modeli Eğit", font="Times 20", activebackground="green",
                  bg="red", fg="white", activeforeground="black",
                  command=modelEgit).place(x=15, y=140)


def fotoCoz():
    img = tf.keras.preprocessing.image.load_img(window.photoPath, target_size=(window.img_height, window.img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    print(window.modelPath)
    model = tf.keras.models.load_model(window.modelPath)
    predictions = model.predict(img_array)

    print(window.predictions[np.argmax(predictions[0], axis=0)])


bDene = tk.Button(opts, text="Fotoğraf Çöz", font="Times 20", activebackground="green",
                  bg="red", fg="white", activeforeground="black",
                  command=fotoCoz).place(x=15, y=220)


def detect_webcam(path):
    try:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(window.img_height, window.img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        print(window.modelPath)
        model = tf.keras.models.load_model(window.modelPath)
        predictions = model.predict(img_array)

        print(window.predictions[np.argmax(predictions[0], axis=0)])
    except:
        print("Predict başarısız!")


def webcamCoz():
    video = cv2.VideoCapture(0)

    while 1:
        _, frame = video.read()
        im = Image.fromarray(frame, "RGB")

        randNum = randrange(1, 999999, 1)

        try:
            if not os.path.exists(window.sourceFolder + "/temp"):
                os.makedirs(window.sourceFolder + "/temp")


        except:
            print("Dosya oluşturulamadı!")

        im.save(os.path.join(window.sourceFolder + "/temp" + "/%i" % randNum + ".jpg"), "JPEG")

        detect_webcam(window.sourceFolder + "/temp" + "/%i" % randNum + ".jpg")

        cv2.imshow("Yok", frame)
        key = cv2.waitKey(1)

        for widget in webcam.winfo_children():
            widget.destroy()

        try:
            img = video.read()[1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_AREA)
            img = ImageTk.PhotoImage(Image.fromarray(img))

            panel2 = tk.Label(webcam, image=img)
            panel2.image = img
            panel2.pack()

        except:
            print("WebCam gösterilemiyor!")

        try:
            os.remove(window.sourceFolder + "/temp" + "/%i" % randNum + ".jpg")
        except:
            print("Böyle bir dosya bulunmamaktadır!")
        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                if not os.path.exists(window.sourceFolder + "/capture"):
                    os.makedirs(window.sourceFolder + "/capture")

                im = np.array(Image.fromarray(frame, 'RGB'))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im, 'RGB')

                im.save(os.path.join(window.sourceFolder + "/capture" + "/%i" % randNum + ".jpg"), "JPEG")

                detect_webcam(window.sourceFolder + "/capture" + "/%i" % randNum + ".jpg")
            except:
                print("Dosya oluşturulamadı!")
    video.release()
    cv2.destroyAllWindows()


bKaydet = tk.Button(opts, text="WebCam Çöz", font="Times 20", activebackground="green",
                    bg="red", fg="white", activeforeground="black",
                    command=webcamCoz).place(x=15, y=300)

bCikis = tk.Button(opts, text="Çıkış", font="Times 20", activebackground="green",
                   bg="red", fg="white", activeforeground="black",
                   command=cikis).place(x=15, y=380)

foo()
window.mainloop()
