#%%
import matplotlib.pyplot as plt
import pandas as pd


font1 = {'family':'serif','color':'blue','size':16}
font2 = {'family':'serif','color':'green','size':12}

#read all Training Data

# LeNet_MaxPool_Kernel_2_Layers with different Kernel number in Layers
filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Size.csv"
df1 = pd.read_csv(filename, sep=";")

filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Number.csv"
df2 = pd.read_csv(filename, sep=";")

# LeNet_MaxPool_Kernel_2_Layers with same Kernel number in Layers (20)
filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Size_2.csv"
df3 = pd.read_csv(filename, sep=";")

filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Number_2.csv"
df4 = pd.read_csv(filename, sep=";")

# LeNet_MaxPool_Kernel_3_Layers with same Kernel number in Layers (20)
filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Size_3_Layers.csv"
df5 = pd.read_csv(filename, sep=";")

filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Number_3_Layers.csv"
df6 = pd.read_csv(filename, sep=";")


# LeNet_MaxPool_Kernel_3_Layers with same Kernel number in Layers (20)
filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Size_3_Layers.csv"
df5 = pd.read_csv(filename, sep=";")

filename = r"Training_Data/Training_Done/LeNet_MaxPool_Kernel_Number_3_Layers.csv"
df6 = pd.read_csv(filename, sep=";")

# BobNets with 2-7  Layers (20)
filename = r"Training_Data/Training_Done/BobNets.csv"
df7 = pd.read_csv(filename, sep=";")

#%% 1
plt.title('LeNet MaxPool / BobNet',fontdict = font2)
plt.plot(df1["Kernel_Size"], df1["Loss"] , label = '2 CNN Layers, Different number of Kernel in layers')
plt.plot(df3["Kernel_Size"], df3["Loss"] , label = 'Same Kernel')
plt.plot(df5["Kernel_Size"], df5["Loss"] , label = 'Same , 3 layers' )
plt.plot(df7["Layers"], df7["Loss"] , label = 'BobNet' )

plt.xlabel('Kernel Size / BobNet Layers',fontdict = font1)
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.legend()
plt.show()


#%%  2
plt.title('LeNet MaxPool',fontdict = font2)
plt.plot(df2["Kernel_Number"],df2["Loss"] , label = 'Different Kernel')
plt.plot(df4["Kernel_Number"],df4["Loss"] , label = 'Same Kernel')
plt.plot(df6["Kernel_Number"],df6["Loss"] , label = 'Same, 3 layers')
plt.xlabel('Kernel Number',fontdict = font1)
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.legend()
plt.show()




#%% 3
plt.title('LeNet MaxPool / BobNet',fontdict = font2)
plt.plot(df1["Kernel_Size"], df1["Accuracy"] , label = 'Different Kernel')
plt.plot(df3["Kernel_Size"], df3["Accuracy"] , label = 'Same Kernel')
plt.plot(df5["Kernel_Size"], df5["Accuracy"] , label = 'Same , 3 layers' )
plt.plot(df7["Layers"],df7["Accuracy"] , label = 'BobNet')
plt.xlabel('Kernel Size / BobNet Layers',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.legend()
plt.show()


#%% 4
plt.title('LeNet MaxPool / BobNet',fontdict = font2)
plt.plot(df2["Kernel_Number"],df2["Accuracy"] , label = 'Different Kernel')
plt.plot(df4["Kernel_Number"],df4["Accuracy"] , label = 'Same Kernel')
plt.plot(df6["Kernel_Number"],df6["Accuracy"] , label = 'Same, 3 layers')
plt.plot(df7["Layers"],df7["Accuracy"] , label = 'BobNet')
plt.xlabel('Kernel Number / BobNet Layers',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.legend()
plt.show()








##################################################################################
##################################################################################
##################################################################################
##################################################################################







#%% LeNet_MaxPool_Kernel_Size different Kernel number in Layers 
df = df1
x  = df["Kernel_Size"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool different Kernel number in Layers ',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Size',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()
#######################################################################################################


#%% LeNet_MaxPool_Kernel_Number different Kernel number in Layers 
df = df2
x  = df["Kernel_Number"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool different Kernel number in Layers ',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Number',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()


#######################################################################################################

#%% LeNet_MaxPool_Kernel_Size with same Kernel number in Layers (20)
df = df3
x  = df["Kernel_Size"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool  same Kernel number in Layers (20)',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Size',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()
#######################################################################################################


#%% LeNet_MaxPool_Kernel_Number with same Kernel number in Layers (20)
df = df4
x  = df["Kernel_Number"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool  same Kernel number in Layers',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Number',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()


#######################################################################################################
#######################################################################################################

#%% LeNet_MaxPool_Kernel_Size_3_Layers with same Kernel number in Layers (20)
df = df5
x  = df["Kernel_Size"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool same Kernel number in Layers (20)',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Size',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()
#######################################################################################################


#%% LeNet_MaxPool_Kernel_Number_3_Layers with same Kernel number in Layers
df = df6
x  = df["Kernel_Number"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('LeNet MaxPool 3 Layers same Kernel number in Layers',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Kernel Number',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()

#######################################################################################################

#%% BobNets
df = df7
x  = df["Layers"]
y1 = df["Loss"]
y2 = df["Accuracy"]

plt.title('BobNets',fontdict = font2)
plt.scatter(x,y1)
plt.plot(x,y1,'r--')
plt.ylabel('Loss',fontdict = font1)
plt.grid(True)
plt.show()
plt.scatter(x,y2)
plt.plot(x,y2,'r--')
plt.xlabel('Layers',fontdict = font1)
plt.ylabel('Accuracy',fontdict = font1)
plt.grid(True)
plt.show()

# %%
