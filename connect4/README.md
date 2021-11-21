# Connect4

This is an homework  developed by Francesco Blangiardi (s288265) for the solution of Connect 4. It was developed with two strategies (MinMax + MCTS and MCTS only) that are briefly described below, along with a simple Human vs AI game handling function.
<br>Both strategies lead to me not being able to win or draw against the ai.

## Montecarlo Only
Montecarlo Tree Search has been implemented in all its 4 stages (Selection, Expansion, Simulation and Backpropagation), and by using a tree structure keeping track of many statistics along with the state of the board.
* Selection happens with a MinMax-like strategy: each node of the tree keeps statistics about the winrate for both players in that sub tree, but at selection time the algorithm will pick nodes with the highest winrate for the **moving player in that state** while going down the tree. This means that whenever we have to pick a leaf to expand, we are going to choose it so that each player has made a (heuristically) good move while reaching that leaf. Additionally nodes that lead to an immediate win for a player or that represent a terminal state will be marked and will not be picked by the selection phase
* Expansion of a node involves generating all its children nodes and simulating each one of them (with consequent Backpropagation of the simulations). However the algorithm treats expansion for nodes leading to a terminal children in a different way: as soon as a terminal children is found (the moving player at the expanded node has a win in 1 move) the node is marked as terminal and all its contribution is backpropagated as complete defeat for the player that reached the expanded state through their previous move. In this way detecting a losing move is less random and the ai should never play or expand one of its children unless it's a forced move
* Simulation is simply a playout of a state through random moves until a terminal state is reached (implementation provided by the professor). Only the number of wins and draws for both players are kept from this stage and they serve as a first heuristic of how good the simulated state is.
* Backpropagation is just the update of the statistics of all the nodes leading to a leaf that has just been expanded

## MinMax + Montecarlo
This strategy is basically a standard MinMax algorithm with alpha-beta pruning that evaluates leaves with MCTS. Unlike the MonteCarlo only strategy (that outputs a winrate), this strategy returns a value in the range (-1,1)

## Conclusions and usage
Both strategies give decent results with somewhat reasonable computing time.
<br>All parameters are in the first cell of the ipynb file and explained through comments. The current parameters are the suggested ones for performing Montecarlo Only evaluation within reasonable time (if you want to switch to MinMax + Montecarlo you should consider lowering the MCTS parameter)
<br>To play against the ai run the last cell of the ipynb file.
<br>If you just want to check the evaluation of a given state you can use the "Example" cell, the encoding of a board is the same as the one suggested by the professor and the function eval_board(board, player) returns the computed best move for a given player along with its evaluation (the evaluation is done as if the selected player is the one who moves)
