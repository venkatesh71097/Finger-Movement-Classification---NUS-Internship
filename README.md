Finger Movement Classification using Convolutional Neural Networks

1. architectures_robio - codes of various CNN based architectures
2. common.py - unpacks the myoband data
3. config.py - sets the number of samples and the gestures to be collected 
4. gui.py - this is for the Pybrain code where I have made the results to be displayed using TKinter's GUI. 
5. myo.py - the main header file for Myoband data. 
6. myo_genData.py - generates the dataset for a subject based on the config.py settings 
7. myo_train.py - trains the dataset based on Pybrain's ANN model. 
8. pred_output.py - utilizes the trained model to test on other subjects when required. 
9. train_testmodel.py - trains the entire collected dataset from trainingsets file and predicts the testing accuracy of the test samples (20%) (It's huge to be transferred. Will do that later)
10. visualize.py - an analysis file that was created to visualize the temporal dataset in the form of images 