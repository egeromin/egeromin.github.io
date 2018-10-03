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
of moves from square to square. After each move he gets to accelerate by +1, 0 or -1 velocity units
in either or both of the X and Y directions, giving a total of 9
possibilities{(+1,+1), (+1, 0), (+1, -1), etc.}. His new velocity then
determines in which square he'll end up next. For safety, let's cap both the
X-speed and Y-speed at 5 units. 

We can write down our task using more formal terminology. Our task is to learn the best possible *action* in the given
*state* we're in. In this case, a state is one of this possible cells in the
grid and at each cell there's 1 of 9 possible actions we can choose from. In
other words, we have to learn an ideal *policy*, an ideal mapping from states
to actions. Again:

- *states* `\(S\)` are defined to be the possible cells in our grid
- *actions* `\(A\)` are defined to be the 9 possible accelerations we can perform at
  each state
- The *policy* is our current strategy for selecting the action given the state
  we're in. This is a mapping `\( \pi: S \to A \)`. Pimi's goal is to learn the
  best possible policy.

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
to completion should intuitively correspond to the best action. Why? Because
the logbooks tell us exactly which of the 9 outcomes has, on average, a more
desirable outcome. If for example logbook 3 corresponding to action `\((+1, -1)\)` averages
87 moves to completion, and logbook 2 corresponding to action `\((+1, 0)\)`
averages 56 moves to completion, then choosing `\((+1, -1)\)` is measurably
better than choosing `\((+1, 0)\)`. 

This intution is in fact correct and has solid theoretical
underpinnings{Barto and Sutton, Chapters 3-5}. 
The particular approach just outlined is an example of a class of
algorithms called *Monte Carlo algorithms* in reinforcement learning. These
algorithms learn the *value* of a particular action `\(a\)` taken at a state
`\(s\)` by running many trials and evaluating the consequence for each trial.
In our case this means letting Pimi race again and again and counting the
number of moves it takes him to complete the track each time. The final *value*
of action `\(a\)` at state `\(s\)` , denoted `\(q(s, a)\)`, is then the 
average length of time to completion after that action, averaged over all of
the trials. We can then use these value estimates to *update* our policy, our
strategy for selecting actions. Pimi takes his logbooks, calculates the best
move at each step and then picks *that* move instead of making a random choice.
The learning problem thus breaks into 2 parts:

- *value estimation*, which is about assigning a numeric score or measure to an
  action at a given state given our current strategy. We want to learn the
  value function `\(q_{\pi}: A \times S \to \mathbb{R}\)`, a function from
  state-action pairs to real numbers. The subscript `\(\pi\)` indicates the
  fact that this function depends on our current policy. This is also known as
  the *prediction* problem, because we are predicting the values.
- *policy iteration*, which is about updating our policy based on our value
  estimates. We update our policy to choose at each stage the best action given
  our current value estimate. This is also known as the *control* problem.

Finding an ideal policy is an iterative process, which involves repeating these
steps over and over again. We start with a random policy and calculate the
values of actions given this behaviour. This tells us which actions are 
optimal -- under our current, *random* behaviour -- and so we modify our policy
to choose instead these better values. A subtle but crucial point here is that
our value estimates are always with respect to our *current policy*, not the
ideal policy. When we perform policy iteration, we move further towards the
ideal policy because we weed out bad decisions: we avoid choosing actions which
our value estimates tell us are measurably bad. But, crucially, our value
estimates don't automatically give us the ideal policy. They merely point us in
the right direction -- *up to a point*. That's why we should always distrust a
bit of our value estimates when updating our policy. Value estimates tell us
how we can improve the policy for short term gain. However, they don't tell us
directly what is the ideal policy. Always going for the short term gain can
preclude long-term, big benefits. For this reason, it's good always to
*explore* when performing policy iteration and make moves different to the ones
pointed to by our value estimates. In practice this means updating our policy
so that it follows the advice given by our value estimates *most* of the time --
this is known as *exploitation* -- but still selecting  a random move
sometimes. This last part is known as *exploration*. The exact proportion
`\(\epsilon\)` of moves that are exploratory is a parameter in our algorithm. 
Different values of `\(\epsilon\)` may lead to different results: a better
policy or faster / slower convergence to the optimum policy. As a rule of
thumb, a higher value of `\(\epsilon\)` can lead to a better policy, but it'll
take longer to find it: slower convergence. This is a tradeoff between
exploration and exploitation and has the fancy name *exploration-exploitation
tradeoff.* 


These are all the components we need to implement our reinforcement learning
algorithm. Here they are again:

- An *initial policy* `\(\pi\)` to start racing. In our case, this is the random policy.
- A method to produce *value estimates* `\(q_\pi\)` for any given policy. These assign a
  score to each action at each state. 
  In our case, this is the Monte Carlo method.
- A way to perform *policy iteration*, in other words to update an existing
  policy `\(\pi\)` and produce an improved policy `\(\pi'\)` using the value
  estimates `\(q_\pi\)`. What we do is a variant of *general policy iteration*, 
  and we update the policy after each *episode*. An *episode* is one run of the
  game: one run of Pimi on the racetrack. And we improve by defining our policy
  to use the action with the *highest* value estimate at each state 90% of the
  time. The remaining 10% of the time we try out random actions to retain a
  healthy amount of exploration. 

So far the policy was a function from states to actions. We can also write it
as a funcion 
`\(\pi(a|s) := \pi(a, s)\)` from state-action pairs to the real numbers. In
this case we interpret the policy
to be a *probability distribution* over `\(a\)` from which we can sample. It's
a probability distribution and so `\(\sum_a \pi(a|s) = 1 \)` for all `\(s\)`.

That's a lot of words. Here's some pseudocode to make things clearer{adapted
from Barto and Sutton, X, Y}. 

1. Choose a small exploration parameter `\(\epsilon\)`, for example 0.1
1. Set the initial policy `\(\pi\)` to be the random policy
1. Initialise our value estimates arbitrarily: set `\(Q(s, a) \in \mathbb{R}\)`
  arbitrarily for all `\(a\)` in `\(A\)`, `\(s\)` in `\(S\)`.
1. `\(\text{Returns}(s, a) \leftarrow \)` empty list, for all `\(a\)` in `\(A\)`, `\(s\)` in `\(S\)`. These *returns* are the entries in our 'success logbooks'.
1. Repeat forever:
    1. Generate an episode according to `\(\pi: S_0, A_0, S_1, A_1 \ldots S_T\)`. This is one run across the racetrack until we hit the green finish line. 
    1. `\(G \leftarrow 0\)`
    1. Loop for each episode, `\(T=t-1, t-2, \ldots 0\)`
        1. `\(G \leftarrow G+1\)`
        1. If the pair `\((S_t, A_t)\)` does **not** appear in `\((S_0, A_0), (S_1, 
           A_1), \ldots (S_{t-1}, A_{t-1})\)`:{this condition gives what's
           called *first-visit Monte Carlo*. This is one variant of Monte Carlo
           prediction. Another is *multiple visits*, which does not have this
           extra 'if' condition.}
           1. Append `\(G\)` to Returns`\((s, a)\)`.
           1. `\(Q(S_t, A_t) \leftarrow \)` average(Returns(`\(S_t, A_t\)`))
           1. `\(A* \leftarrow \text{arg max}_a Q(S_t, a)\)`. Break ties
              arbitrarily.
           1. For all `\(a \in A(S_t)\)`, where `\(A(S_t)\)` is the set of
              possible actions we can take at `\(S_t\)`:

              ```
              \[
              \pi(a|S_t) \leftarrow = \begin{cases}
                  1 - \epsilon + \epsilon / |A(S_t)| & \text{if } a = A* \\
                  \epsilon / |A(S_t)| & \text{otherwise}
              \end{cases}
              \]
              ```

I've implemented a Monte Carlo algorithm for the racetrack and here are my
results:

INSERT IMAGE

The implementation is on [GitHub](bla) and I encourage you to take a look.
Let's zoom in and take a closer look at the most important parts.

