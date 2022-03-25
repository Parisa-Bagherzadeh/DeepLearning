# Assignment42
  <strong> Using ArcFace network to get the weights and train a face recognition model </strong></br>
  There is 75 data ,51 data to train and 23 data to test the model</br>
  Model classifies 4 classes
  
   * 100% accuracy on testing data

  <img src='https://github.com/Parisa-Bagherzadeh/DeepLearning/blob/main/Assignment42/charts/output.png' >
  
# Assignment43

  <strong>Multi Layer Perceptron and Deep learning using keras on four datasets</strong>
 
<table>
  <tr><td></td><td colspan=2><strong>Accuracy</strong></td></tr>
  <tr><td><strong>Dataset</bold></td><td><bold>MLP</strong></td><td><bold>Deep Learning</bold></td></tr>
  <tr><td>mnist</td><td>0.96</td><td>0.95</td></tr>
  <tr><td>fashion_mnist</td><td>0.88</td><td>0.81</td></tr>
  <tr><td>cfar10</td><td>0.40</td><td> 0.69</td></tr>
  <tr><td>cfar100</td><td>0.15</td><td>0.26</td></tr>
</table>

# Assignment44

 <strong>Classification model on four datasets</strong>
  
  This classification model classifies 4 objects
   
  * Car   🚗
  * Dress 👗
  * House 🏠
  * Pizza 🍕

   
   <table>
     <tr>
       <td></td>
       <td>Accuracy</td>
       <td>Loss</td>
     </tr>
     <tr>
       <td>Train</td>
       <td>0.79</td>
       <td>0.54</td>
     </tr>
     <tr>
       <td>Validation</td>
       <td>0.72</td>
       <td>0.67</td>
     </tr>
     <tr>
       <td>Test</td>
       <td>0.86</td>
       <td>0.34</td>
     </tr>
   </table>

# Detecting normal people and sheykh people

   A deep learning model to detect normal and kheykh people,</br>
   also you can use this telegram bot <a href='https://t.me/parisabagherzadeh_bot'>@parisabagherzadeh_bot</a>  to do the job</br>
   Just send the bot a picture of a sheykh or a normal person</br>

   <strong>The model used here is VGG16</strong></br>

   <table>
     <tr>
       <td></td>
       <td>Accuracy</td>
       <td>Loss</td>
     </tr>
     <tr>
        <td>Train</td>
        <td>0.99</td>
        <td>0.01</td>
     </tr>
     <tr>
        <td>Validation</td>
        <td>0.98</td>
        <td>0.13</td>
     </tr>
     <tr>
        <td>Test</td>
        <td>0.97</td>
        <td>0.12</td>
     </tr>
   </table>
  

# 17 Flower classification
  
  A deep learning model using VGG16 convolution neural net is trained to classify flowers

  <table>
     <tr>
       <td></td>
       <td>Accuracy</td>
       <td>Loss</td>
     </tr>
     <tr>
        <td>Train</td>
        <td>0.98</td>
        <td>0.06</td>
     </tr>
     <tr>
        <td>Validation</td>
        <td>0.90</td>
        <td>0.48</td>
     </tr>
     <tr>
        <td>Test</td>
        <td>0.82</td>
        <td>1.06</td>
     </tr>
   </table>
  
  

# Real time face mask  detector 
  
This is a model based on vgg16 which detects face mask in real time</br>
To train the model this dataset https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset is used </br>
To use the model :</br>

* first save the model from Face_Mask_Detection.ipynb file</br>
* then run inference.py file

<table>
     <tr>
       <td></td>
       <td>Accuracy</td>
       <td>Loss</td>
     </tr>
     <tr>
        <td>Train</td>
        <td>0.99</td>
        <td>0.01</td>
     </tr>
     <tr>
        <td>Validation</td>
        <td>0.98</td>
        <td>0.55</td>
     </tr>
     <tr>
        <td>Test</td>
        <td>0.98</td>
        <td>0.35</td>
     </tr>
   </table>
  
  
# Age estimation

  Automatioc human age estimation based on human facial appearance ,</br>
  using ResNet50 convolutional neural network</br>
  After saving this model from AgePrediction.ipynb file ,</br>
  use the command below in AgePrediction.py file for estimating the age :</br>

  python3 AgePrediction.py --input ./input --output ./output
 


