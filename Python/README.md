This is the readme file for this project. The project is built on python3.7.3

This python3 file operates like a normal python file. Once in the correct source folder, type "python3 postagging.py" to run the script.

The leave-one-out method can be done by editing the "selected_attributes" variable to the attributes that are selected.

I imported warnings to only display warnings once, since there are a few repeated errors for my implementation due to an error with some labels in y_test not appearing in y_pred, so I tried to add a np.unique lable to it also.