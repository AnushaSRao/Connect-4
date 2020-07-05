# Connect-4
Connect-4 is a two-player board game in which players take turns dropping one
colored disc from the top into a seven-column, six-row vertically suspended grid.
The pieces fall straight down, occupying the lowest available space within the
column. The objective of the game is to be the first to form a horizontal, vertical,
or diagonal line of four of one's own discs. One measure of complexity of the
Connect Four game is the number of possible games board positions. For classic
Connect Four played on 6 rows, 7 columns grids, there are 4,531,985,219,092
positionsfor all game boards populated with 0 to 42 pieces.<br>
The solution uses Q-learning, which is an algorithm based on Reinforcement
Learning to train the robot to play against a human. This method is about taking
suitable action to maximize reward in a particular situation. Reinforcement
learning differs from the supervised learning in a way that in supervised learning
the training data has the answer key with it so the model is trained with the correct
answer itself whereas in reinforcement learning, there is no answer but the
reinforcement agent decides what to do to perform the given task. In the absence of
a training dataset, it is bound to learn from its experience. <br>
The problem has been approached in two ways - Tensorflow, Keras. Each approach
being followed has an unique reward function and a different way of representing
the board. In addition, the Tensorflow approach has a deeper Neural Network as
compared to the Keras. <br> 
The implemented game has 3 modes - <br>
● Single Player <br>
● Two Player <br>
● AI vs AI
