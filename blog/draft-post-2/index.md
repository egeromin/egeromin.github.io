# TITLE

- learning from experience
- driver first, using monte carlo methods
- tic tac toe
- driver
- when there are too many states: give some examples, including mastermind


-- nice intro--


Let's get up to speed with an example: racetrack driving{taken from Sutton and
Barto, *reinforcement learning*, Chapter X, etc}. We'll take the famous Formula 1 
racing driver Pimi Roverlainen and transplant him onto a racetrack in gridworld.

INSERT IMAGE

Pimi's task is to learn how to drive from any given point on the red starting line to any
point of his choice on the green ending line. He should do this as quickly as
possible. We'll model Pimi's driving in a simplified way: namely as a sequence
of moves from square to square{a *Markov Decision Process* by our earlier
terminology}. After each move he gets to accelerate by +1, 0 or -1 velocity units
in either or both of the X and Y directions, giving a total of 9
possibilities{(+1,+1), (+1, 0), (+1, -1), etc.}. His new velocity then
determines in which square he'll end up next. For safety, let's cap both the
X-speed and Y-speed at 5 units. 

On such an easy course, this is an easy task for Pimi. So, to make things harder, 
we'll blindfold him so he cannot see where he's going. All he has access to is
9 *success logbooks* at each square in the racetrack -- so `\(9N\)` logbooks,
where `\(N\)` is the number of squares in the track. Each logbook belongs to 1 of
the 9 possible accelerations Pimi could make, say for example `\((+1, 0)\)`. Then
this logbook `\((+1, 0)\)` has a 
 full record of the number of moves it's taken Pimi to
arrive at the green line in the past starting at the current square, given
that the action taken was `\((+1, 0)\)`. 

As Pimi drives with his blindfold on, he drives at random by accelerating in
random directions because he can't see where he's going. He often crashes and
when this happens, he takes a sip of refreshing blueberry juice, fetches a
plaster or two and then starts again from a random point on the red starting line,
keeping a stiff upper lip. As he does so, he crosses more and more squares, the
smoke rises, the racetrack gets worn and the `\(9N\)` logbooks fill up. 
That's a lot of paper! 

Once there's a fair amount of records in each of the `\(9N\)` logbooks, Pimi can
start using them to make decisions instead of driving and crashing randomly.
Each success logbook is indeed a measure of success of its corresponding action at that
square in gridworld: the logbook averaging the *lowest* number of moves
to completion should intuitively correspond to the best action. In fact, this
is the case, and the approach just outlined is an example of a class of
algorithms called *Monte-Carlo algorithms* in reinforcement learning. 


