# LSTM

Helpful blogs while writing notes here:

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## RNN Structure
Term define: batch_input_shap = (nb_samples, timesteps, nb_features)

**nb_samples** is the number of samples available in the training data set, namely how many sequences being used. **timesteps** is the length of one individual sample sequence. **nb_features** is the dimension of input features, the length of vector that we input to RNN at each timestep recursively.

Basic RNN structure has the hidden states going through time steps and takes in input at each time step. 

**h<sub>t**: Hidden states
  
**x<sub>t**: Input vector at time step t
  
**o<sub>t**: Output vector at time step t
  
**U**: Input weight vector

**W**: States weight vector

**V**: Output weight vector
  
The hidden states are updated at each time step by combining hidden states from last step and input at current step, 

<a href="https://www.codecogs.com/eqnedit.php?latex=h_t&space;=&space;tanh(U\cdot&space;x_t&space;&plus;&space;W\cdot&space;h_{t-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_t&space;=&space;tanh(U\cdot&space;x_t&space;&plus;&space;W\cdot&space;h_{t-1})" title="h_t = tanh(U\cdot x_t + W\cdot h_{t-1})" /></a>

[\\]: # (This is a comment)

RNN can output one result **o<sub>t** at each time step by multiplying the hidden states at current step with the output weight vector **V** (an additional activation function can be applied to **h<sub>t** afterwards),
  
<a href="https://www.codecogs.com/eqnedit.php?latex=o_t&space;=&space;V\cdot&space;h_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o_t&space;=&space;V\cdot&space;h_t" title="o_t = V\cdot h_t" /></a>

Depending on particular situations, the output of RNN can be eigher a single result (one value or one vector) or multiple results. 

## LSTM structure

LSTM mitigates the problem of vanishing gredient that ordinary RNN suffers from by introducing the gating mechanism that allows LSTM to learn long-term dependencies. LSTM is still a RNN but with better design to pass the states over timesteps. Besides the hidden states **h<sub>t**, there is another cell states **C<sub>t** going through insdie LSTM, which can be considered as the internal memory of the neural network. There are three special gates in LSTM structure, 

-Input gate: <a href="https://www.codecogs.com/eqnedit.php?latex=i&space;=&space;sigmoid(U^i\cdot&space;x_t&space;&plus;&space;W^i\cdot&space;h_{t-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i&space;=&space;sigmoid(U^i\cdot&space;x_t&space;&plus;&space;W^i\cdot&space;h_{t-1})" title="i = sigmoid(U^i\cdot x_t + W^i\cdot h_{t-1})" /></a>

-Forget gate: <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;=&space;sigmoid(U^f\cdot&space;x_t&space;&plus;&space;W^f\cdot&space;h_{t-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;=&space;sigmoid(U^f\cdot&space;x_t&space;&plus;&space;W^f\cdot&space;h_{t-1})" title="f = sigmoid(U^f\cdot x_t + W^f\cdot h_{t-1})" /></a>

-Output gate: <a href="https://www.codecogs.com/eqnedit.php?latex=o&space;=&space;sigmoid(U^o\cdot&space;x_t&space;&plus;&space;W^o\cdot&space;h_{t-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o&space;=&space;sigmoid(U^o\cdot&space;x_t&space;&plus;&space;W^o\cdot&space;h_{t-1})" title="o = sigmoid(U^o\cdot x_t + W^o\cdot h_{t-1})" /></a>

Basically the three gates are calculated based on the hidden states at last time step and input at current time step in the same way but with different sets of weight vectors **U** and **W**. According to the name of the gates, input gate decides how much information is allowed to go through from the input at current time step, forget gate decides how much information from previous timesteps gets through while updating the cell states **C<sub>t**, output gate decides the output from current states. The cell states are updated every time step by first generating a cell states candidate based on the input and hidden states from last recurrence, 
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{C_t}&space;=&space;tanh(U_c&space;\cdot&space;x_t&space;&plus;&space;W_c&space;\cdot&space;h_{t-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{C_t}&space;=&space;tanh(U_c&space;\cdot&space;x_t&space;&plus;&space;W_c&space;\cdot&space;h_{t-1})" title="\tilde{C_t} = tanh(U_c \cdot x_t + W_c \cdot h_{t-1})" /></a>

Then the cell states at current time step are updated with the candidate cell states and the cell states from last recurrence, 

<a href="https://www.codecogs.com/eqnedit.php?latex=C_t&space;=&space;i&space;\cdot&space;\tilde{C_t}&space;&plus;&space;f&space;\cdot&space;C_{t-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_t&space;=&space;i&space;\cdot&space;\tilde{C_t}&space;&plus;&space;f&space;\cdot&space;C_{t-1}" title="C_t = i \cdot \tilde{C_t} + f \cdot C_{t-1}" /></a>

Then hidden states at current time step are calculated based on **C<sub>t** and the output gate o, 
  
<a href="https://www.codecogs.com/eqnedit.php?latex=h_t&space;=&space;o&space;\cdot&space;tanh(C_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_t&space;=&space;o&space;\cdot&space;tanh(C_t)" title="h_t = o \cdot tanh(C_t)" /></a>

The output at each time step is obtained by multiplying **h<sub>t** with another weight matrix, 
  
<a href="https://www.codecogs.com/eqnedit.php?latex=o_t&space;=&space;V&space;\cdot&space;h_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o_t&space;=&space;V&space;\cdot&space;h_t" title="o_t = V \cdot h_t" /></a>

## Stateful
- In Keras, set stateful=True in the fit() function allows the model to keep the states over training iteration unless reset manually. Stateful=True usually comes with shuffle=False so that the dependencies between samples can be utilized, passing through by the ongoing state. (http://philipperemy.github.io/keras-stateful-lstm/)
- In stateless mode, the states are reset after each batch and there are batch_size states in paralell while training on a batch, meaning the there are 16 states being initialized, updated, and reset during one batch training if batch_size = 16. Therefore the states in stateless mode are totally independent from each other on a sample base. (https://stackoverflow.com/questions/43882796/when-does-keras-reset-an-lstm-state)
- Batch_size must be able to divide nb_samples in stateful mode because the states frome last batch are transferred to the next batch. (https://datascience.stackexchange.com/questions/32831/batch-size-of-stateful-lstm-in-keras)
