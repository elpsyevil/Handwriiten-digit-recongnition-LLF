# HANDWRITTEN DIGIT RECOGNITION USING LOCAL LINE FITTING

<div class="cell markdown">

# Imports

</div>

<div class="cell code" data-execution_count="1">

``` python
import time
import timeit
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import tkinter.font as tkFont
import torch 
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tkinter import *
from PIL import ImageGrab, Image, ImageOps, EpsImagePlugin
from skimage.io import imread,imshow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

```

</div>

<div class="cell markdown">

# Function definitions

</div>

<div class="cell code" data-execution_count="8">

``` python
p = transforms.Compose([transforms.Resize((128,128))])

def resize_image(img):
    
    img = p(img)
    img = img.convert('L')
    img = np.array(img)
    img = img/255.0
    return img

def crop_image(img, tol=0):
    
    mask = img <= 0
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    cropped = img[x0:x1, y0:y1]
    return cropped

def slice_img(imarr):
    l= []
    for i in range(0, len(imarr), int(len(imarr)/4)):
        for j in range(0, len(imarr), int(len(imarr)/4)):
            l.append(imarr[0+i: int(len(imarr)/4)+i, 0+j: int(len(imarr)/4)+j])
    return l

def nb_pixels(block):
    i=0.0
    for a in block:
        for b in a :
            if b!=1.0:
                i += 1.0
    return i

def linReg(b):
    x=[]
    y=[]
    for i in range(len(b)):
        for j in range(len(b)):
            if b[i,j]!=1.0:
                x.append(i)
                y.append(j)
    x.reverse()
    if nb_pixels(b)!=0.0:
        model = LinearRegression().fit(np.array(x).reshape((-1, 1)), np.array(y))
        return (model.coef_[0], model.intercept_)
    else : 
        return (0.0, 0.0)

def feature_extraction(img):
    plt.imsave('ima.png', img, cmap='gray')
    img1 = Image.open('ima.png').convert('RGB')
    img1 = ImageOps.invert(img1)
    imarr = resize_image(img1)

    for i in range(len(imarr)) :
        for j in range(len(imarr[i])):
            if imarr[i][j]!=1.0:
                imarr[i][j]=0.0

    imarr = crop_image(imarr)

    plt.imsave('cropped.png', imarr,cmap='gray')

    cropped_image = Image.open('cropped.png')

    crop_arr = resize_image(cropped_image)

    for i in range(len(crop_arr)) :
        for j in range(len(crop_arr[i])):
            if crop_arr[i][j]!=1.0:
                crop_arr[i][j] = 0.0

    l = slice_img(crop_arr)
    n = 0.0
    for b in l :
        n += nb_pixels(b)
    a = []
    for b in l :
        a.append(nb_pixels(b)/n)
        a.append((2*linReg(b)[1])/(1+linReg(b)[1]**2))
        a.append((1-linReg(b)[1]**2)/(1+linReg(b)[1]**2))
    a = torch.tensor(a)
    return a
```

</div>

<div class="cell markdown">

# Feature extraction

</div>

<div class="cell code" data-execution_count="2">

``` python


training_dataset = datasets.MNIST(root='./data' ,train=True,download=True,transform=None)
test_dataset = datasets.MNIST(root='./data' ,train=False,download=True,transform=None)

donnees_apprentissage,donnees_validation = train_test_split(training_dataset.data.numpy(), test_size=0.16, random_state=42)
label_apprentissage,label_validation = train_test_split(training_dataset.targets.numpy(), test_size=0.16, random_state=42)
label_test = test_dataset.targets

fe_app = []
for i in range(len(donnees_apprentissage)):
    start = timeit.default_timer()

    fe_app.append(feature_extraction(donnees_apprentissage[i]))
    stop = timeit.default_timer()
    time_left = (stop-start)/60
    sys.stdout.write('\r'+str(i+1)+'/'+str(len(donnees_apprentissage))+'  :  Temps restant : '+str(time_left*len(donnees_apprentissage)-i*time_left)+' min')
    sys.stdout.flush()


fe_val = []
for i in range(len(donnees_validation)):

    start = timeit.default_timer()
    fe_val.append(feature_extraction(donnees_validation[i]))
    stop = timeit.default_timer()
    time_left = (stop-start)/60
    sys.stdout.write('\r'+str(i+1)+'/'+str(len(donnees_validation))+'  :  Temps restant : '+str(time_left*len(donnees_validation)-i*time_left)+' min')
    sys.stdout.flush()

fe_test = []
for i in range(len(test_dataset.data)):
    start = timeit.default_timer()
    fe_test.append(feature_extraction(test_dataset.data[i]))
    stop = timeit.default_timer()
    time_left = (stop-start)/60
    sys.stdout.write('\r'+str(i+1)+'/'+str(len(test_dataset.data))+'  :  Temps restant : '+str(time_left*len(test_dataset.data)-i*time_left)+' min')
    sys.stdout.flush()
```

<div class="output stream stdout">

    10000/10000  :  Temps restant : 0.002262590000007947 minn

</div>

</div>

<div class="cell markdown">

# Features saving

</div>

<div class="cell code" data-execution_count="4">

``` python
import pickle
with open("features_app", "wb") as file:
    pickle.dump(fe_app, file)
    
with open("features_val", "wb") as file:
    pickle.dump(fe_val, file)

with open("features_test", "wb") as file:
    pickle.dump(fe_test, file)
    
with open("targets_app", "wb") as file:
    pickle.dump(label_apprentissage, file)

with open("targets_val", "wb") as file:
    pickle.dump(label_validation, file)
```

</div>

<div class="cell markdown">

# Learning

</div>

<div class="cell code">

``` python
class MyDataset(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __getitem__(self,idx):
        return (self.data[idx],self.targets[idx])
    def __len__(self):  
        return len(self.targets)

my_training_dataset = MyDataset(fe_app,label_apprentissage)
my_test_dataset = MyDataset(fe_test,label_test)
my_validation_dataset = MyDataset(fe_val,label_validation)

BATCH_SIZE = 10

train_loader = torch.utils.data.DataLoader(my_training_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(my_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(my_validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

class myNN(nn.Module):
    def __init__(self):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(48, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

myModel = myNN()

loss = nn.CrossEntropyLoss()
opt = optim.Adam(myModel.parameters(), lr=0.001)
n_epochs = 100

from torch.autograd import Variable
for epoch in range(n_epochs):
    myModel.train()
    
    t_cost= 0.0
    
    for i,(inputs,labels) in enumerate(train_loader):
        inputs = inputs.float()
        labels = labels.float()
        outputs = myModel(inputs)
        cout = loss(outputs,labels.long())
        # Backpropagation: 
        # Réinitialiser l'optimiseur
        opt.zero_grad()
        cout.backward()
        opt.step()
        t_cost += cout
    t_cout_moy = t_cost/(len(train_loader))
    v_cost= 0.0
    n_prev = 0
    myModel.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_losss =0
        for i, data in enumerate(validation_loader):    
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            outputs = myModel(inputs)
            cout = loss(outputs,labels.long())
            v_cost += cout
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
        v_cout_moy = v_cost/(len(train_loader))
        accuracy = correct / total
        print(accuracy)
c_test = 0.0
n_prev_c = 0
with torch.no_grad():        
    for i,(inputs,labels) in enumerate(test_loader):        
            inputs = inputs.float()
            labels = labels.float()
            outputs = myModel(inputs)
            cout = loss(outputs,labels.long())
            c_test += cout
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

    c_test_moy = c_test/len(train_loader)


    n_prev_cmoy = correct/total
    print('test : '+ str(n_prev_cmoy))
```

</div>

<div class="cell markdown">

# GUI

</div>

<div class="cell code" data-execution_count="11">

``` python
EpsImagePlugin.gs_windows_binary =  r'C:\Program Files\gs\gs9.53.3\bin\gswin64c.exe'

class main:
    def __init__(self,master):
        self.master = master
        self.text = StringVar()
        self.text.set('')
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 15
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)
        frame = Frame(master)
        frame.pack()
        self.button = Button(frame,text="PREDICT",fg="red",command=self.predict)
        self.button.pack(side=LEFT)
        self.buttonclear = Button(frame,text="CLEAR",fg="red",command=self.clear)
        self.buttonclear.pack(side=LEFT)
        self.fontStyle =  tkFont.Font(family="Lucida Grande", size=40)
        self.label = Label(master, textvariable=self.text,font=self.fontStyle)

        self.label.pack()


    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,
                               self.old_y,
                               e.x,
                               e.y,
                               width=self.penwidth,
                               fill=self.color_fg,
                               capstyle=ROUND,
                               smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    
        self.old_x = None
        self.old_y = None      

    def clear(self):
        self.c.delete(ALL)

    def drawWidgets(self):

        self.c = Canvas(self.master,width=500,height=500,bg=self.color_bg,highlightthickness=0)
        self.c.pack(expand=False)


    def predict(self) :
        HWND = self.c.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        with torch.no_grad():
            output = myModel(feature_extraction(resize_image(im)).float())

        self.text.set(str(int(torch.argmax(output))))

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()
```

</div>

<div class="cell markdown">

# Model saving

</div>

<div class="cell code">

``` python
torch.save(myModel.state_dict(), 'leModel100')
```

</div>

<div class="cell markdown">

# Model loading

</div>

<div class="cell code" data-execution_count="4">

``` python
class myNN(nn.Module):
    def __init__(self):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(48, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

myModel=myNN()
myModel.load_state_dict(torch.load('leModel100'))
```

<div class="output execute_result" data-execution_count="4">

    <All keys matched successfully>

</div>

</div>

<div class="cell code">

``` python
```

</div>
