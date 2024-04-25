# AI-Sentiment-Analysis with Tensorflow <br>
1. Setup Project
2. Project Describtion
3. Code Describtion
<br>
# 1. Setup the project and try if your sentence is sarcastic or not<br>
1. Clone the repository and run it with visual studio code.<br>
2. In the code file change variable "json_datei" to link to the json file "sarcasm.json".<br> json_datei = r'YOUR_LINK\sarcasm.json' . This is the dataset of 26.000 headlines.<br>
3. Run the script<br>
4. When the Plots are poping up you have to close them. Then the code will go on and show outputs in the console<br>
5. Scroll down to the last lines in the code. There is the sentence variable. Type in a sentence to find out if the trained model think it is sarcastic or not. The Output values ins the console are between 0.1, 0.2... to 1. The number one is sarcastic and the nnumber zero is not sarcastic.

# 2. Project Description<br>
This project utilizes TensorFlow and Keras to build a neural network for detecting sarcasm in headlines. The dataset, containing headlines and corresponding sarcasm labels, is loaded and preprocessed using tokenization and padding techniques. A neural network model is constructed with an embedding layer followed by global average pooling and dense layers. The model is trained on the dataset to achieve accurate sarcasm detection and its performance is visualized through accuracy and loss graphs. Additionally, word embeddings are extracted and visualized for further analysis.


# 3. Short description of the code. The code file is commented as well:
The code begins by importing necessary libraries such as json for handling JSON data and tensorflow for building and training neural networks. The script defines parameters such as vocabulary size, embedding dimension, and maximum sequence length. It then loads a dataset of 26000 headlines and their sarcasm labels from a JSON file, splits the data into training and testing sets and tokenizes the text data while converting it into padded sequences for input into the neural network.
The neural network model is built using Keras's Sequential API. It consists of an embedding layer to convert words into dense vectors, followed by a global average pooling layer to reduce dimensionality.  The model is compiled with binary cross entropy loss and the Adam optimizer. The model is trained on the training data for 30 epochs with validation performed on the testing data. The training history, including accuracy and loss metrics, is stored for visualization. The script includes a function to plot accuracy and loss graphs based on the training history. The trained model's embedding layer is used to extract word embeddings. They are then saved to files for visualization in a separate tool TensorFlow Embedding Projector. Finally, the model is used to predict sarcasm in new sentences provided in the script. The sentences are tokenized, padded, and fed into the model for prediction.
Overall, the code preprocesses text data, builds and trains a neural network model, visualizes training metrics, extracts word embeddings and performs predictions on new data.

