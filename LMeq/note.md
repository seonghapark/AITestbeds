# What I did and received comments:

### Jun 19 @ research meeting
1. This is another library - https://github.com/jax-ml/jax 

2. Honestly, for the model part, I would just try some simple examples in ChatGPT or Claude to see how it does. I guess it would help me to know exactly how the equation part is expressed but I’ve had good luck with a simple zero shot prompting approach like:
```
Your job is to take a user description of an optimization problem and translate it into a mathematical Python expression which can be solved using scipy's fsolve. Here are a few examples:

INPUT: I have two devices. One has 4 CPUs and the other has 2, please optimize my workload across the two.
OUTPUT: L(x, y, a) 2*x + y + a(x + y - 1) + ...

INPUT: ...
OUTPUT: ...

Now, here is the user's request. Please output the right expression:
{insert user prompt here}
```
Admittedly, this is really high level, since I don’t really know what problem you all are working on… but it’s a general approach I’ve used before.

As for the solver, definitely look at scipy’s optimization and solving components: https://docs.scipy.org/doc/scipy/reference/optimize.html
I used to use fsolve as a general numeric solver. There’s also optimization type functions in there too like minimize. (edited) 

3. The things that YH delivered that the two mentioned at the meeting: a. fine-tuning b. give 30-50 pages of information about optimizer and problem, and let a language model outputs an optimization equation, c. use optimizers in ECP (Exo-scale computing project) and find the best match.

4. The Sameer Shende at University of Oregon ([and ASCR?](https://www.linkedin.com/in/SameerShende)) is also interested in the same topic. Are you interested in working with his team? **Which actually means** why don't you work with his team.
