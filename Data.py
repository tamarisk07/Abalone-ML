import pandas as pd

#######################
# import .data file and read it
# from 'Sex' to 'Shell weight' gives the 8 attributes while 'Rings'+1.5 gives the age(year) of the abalones
data = pd.read_table('~/Downloads/abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'], sep=',')

#######################
# convert Letter 'M''F''I' to value 1, 0.5, 0 respectively
for i in range(4177):
    if data.iloc[i, 0] == 'M':
        data.iloc[i, 0] = 1
    if data.iloc[i, 0] == 'F':
        data.iloc[i, 0] = 0.5
    if data.iloc[i, 0] == 'I':
        data.iloc[i, 0] = 0

# changed 'Rings' to 'Age' by adding 1.5 to the previous value
data['Rings'] += 1.5

#######################
# distract X and Y_real from  the data_frame
X = data[['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']]
Y_real = data['Rings']
