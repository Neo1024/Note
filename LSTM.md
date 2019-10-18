# LSTM

Helpful blogs while writing notes here:

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## RNN Structure
Basic RNN structure has the hidden states going through time steps and takes in input at each time step. 

**s<sub>t**: Hidden states
  
**x<sub>t**: Input vector at time step t
  
**h<sub>t**: Output vector at time step t
  
**U**: Input weight vector

**W**: States weight vector

**V**: Output weight vector
  
The hidden states are updated at each time step by combining hidden states from last step and input at current step, 

<a href="https://www.codecogs.com/eqnedit.php?latex=s_t&space;=&space;softmax(U\cdot&space;x_t&space;&plus;&space;W\cdot&space;s_{t-1})" target="_blank"><img src="https://latex.codecogs.com/png.latex?s_t&space;=&space;softmax(U\cdot&space;x_t&space;&plus;&space;W\cdot&space;s_{t-1})" title="s_t = softmax(U\cdot x_t + W\cdot s_{t-1})" /></a>



## Stateful
- Term define: batch_input_shap = (nb_samples, look_back/stepsize, nb_features)
- In Keras, set stateful=True in the fit() function allows the model to keep the states over training iteration unless reset manually. Stateful=True usually comes with shuffle=False so that the dependencies between samples can be utilized, passing through by the ongoing state. (http://philipperemy.github.io/keras-stateful-lstm/)
- In stateless mode, the states are reset after each batch and there are batch_size states in paralell while training on a batch, meaning the there are 16 states being initialized, updated, and reset during one batch training if batch_size = 16. Therefore the states in stateless mode are totally independent from each other on a sample base. (https://stackoverflow.com/questions/43882796/when-does-keras-reset-an-lstm-state)
- Batch_size must be able to divide nb_samples in stateful mode because the states frome last batch are transferred to the next batch. (https://datascience.stackexchange.com/questions/32831/batch-size-of-stateful-lstm-in-keras)
