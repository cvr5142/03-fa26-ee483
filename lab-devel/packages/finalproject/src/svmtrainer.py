#!/usr/bin/env python3

# import numpy as np
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler 
# from sklearn.svm import SVC 
# from sklearn.metrics import accuracy_score, classification_report
#import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from skimage.transform import resize
from skimage.io import imread

# # Loading the Dataset
# data = load_xxxx_xxxx()
# X = data.data[:, [0,1]] 
# y = data.target 

# # Splitting the Data
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) 

# # Scale the Features
# scaler = StandardScaler() 
# X_train_scaled = scaler.fit_transform(X_train) 
# X_test_scaled = scaler.transform(X_test) 

# # Train the SVM Classifier
# svm_classifier = SVC(kernel='linear', C=1.0, random_state=42) 
# svm_classifier.fit(X_train_scaled, y_train) 

# # Evaluate the Model
# y_pred = svm_classifier.predict(X_test_scaled) 
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print(classification_report(y_test, y_pred, target_names=data.target_names)) 

# # Visualizing the Decision Boundary
# def plot_decision_boundary(X, y, model, scaler):
#     h = 0.02 #Step size for mesh
#     x_min, x_max = X[:,0].min() - 1, X[:,0].max + 1
#     y_min, y_max = X[:,0].min() - 1, X[:,0].max + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     # Predict on mesh points
#     Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
#     Z = Z.reshape(xx.shape)

#     # Plot decision boundary and data points
#     plot.contour(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
#     plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors+'k')
#     plt.xlabel(data.feature_names[0])
#     plt.ylabel(data.feature_names[1])
#     plt.title('SVM Decision Boundary')
#     plt.show()
# plot_decision_boundary(X_train, y_train, svm_classifier, scaler)x_min, x_max = X[:,0].min() - 1, X[:,0].max + 1

# Load Images and Convert to DataFrame
Categories=['Stop Sign','Not Stop Sign']
flat_data_arr=[] 
target_arr=[] 
datadir='/code/catkin_ws/src/lab-devel/packages/finalproject/src/IMAGES/' 


for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        full_img_path =(os.path.join(path,img))
        if not os.path.isfile(full_img_path): continue #I added this not sure if it is needed
        img_array=imread(full_img_path)
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

#flat_data=np.array(flat_data_arr)
#target=np.array(target_arr)
x=np.array(flat_data_arr)
y=np.array(target_arr)

#df=pd.DataFrame(flat_data)
#df['Target']=target        These are pd ones i think
#print(df.shape)

# Separate Input Features and Targets
#x=df.iloc[:,:-1]  ichanged these too turned them into comments
#y=df.iloc[:,-1] 

# Train-Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

# Build and Train the Model
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

svc=svm.SVC(probability=True)

model=GridSearchCV(svc,param_grid)

model.fit(x_train, y_train)

# Model Evaluation
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)

print(f"The model is {accuracy*100}% accurate")
print(classification_report(y_test, y_pred, target_names=['Not Stop Sign', 'Stop Sign']))

# Prediction 
path='/home/student/03-fa26-ee483/lab-devel/finalproject/src/IMAGES/Test/0P2QYA1QVLWI.jpg'

if os.path.exists(path): #i added this line
    img=imread(path)
    plt.imshow(img)
    plt.show()

img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]

probability=model.predict_proba(l)

for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
    
print("The predicted image is : "+Categories[model.predict(l)[0]])