# What did I do:

## Comments from Waggle people:
### Jun 19 @ research meeting
1. This is another library - https://github.com/jax-ml/jax **--> this is a lib that google people developed in 2018, and seems both input and output are equation? But that is not what I am trying to do, so reject this.**

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
Admittedly, this is really high level, since I don’t really know what problem you all are working on… but it’s a general approach I’ve used before. **--> Using GhatGPT, fine, for a simple trial and example, but not for demonstrating Henry's optimzation model and writing a paper. So reject this too.**

As for the solver, definitely look at scipy’s optimization and solving components: https://docs.scipy.org/doc/scipy/reference/optimize.html
I used to use fsolve as a general numeric solver. There’s also optimization type functions in there too like minimize. (edited) **--> Solvers, sure thing. I may find things from scipy and other libs, and can make a system to find the best optimzation equation, but may need to compare their performance with Henry's.**

3. The things that YH delivered that the two mentioned at the meeting: a. fine-tuning b. give 30-50 pages of information about optimizer and problem, and let a language model outputs an optimization equation, c. use optimizers in ECP (Exo-scale computing project) and find the best match. **--> I don't know if I need to fine-tune or whatever, but will try with LLaVA or another model, then may need to make a system to run `ollama.cpp`, run a language model, and test the language model like what Sean did on ChatGPT. Or I don't need to run `ollama.cpp` but just standalone app? And I maybe need to implement optimizers in ECP, after some testing use of LMs**

4. The Sameer Shende at University of Oregon ([and ASCR?](https://www.linkedin.com/in/SameerShende)) is also interested in the same topic. Are you interested in working with his team? **Which actually means** why don't you work with his team.


## And so with Sophia:
Well, I will start with some complaints. Waggle people changes IP address from time to time, and never let me know!!! That drives me mad at them!!! (and somewhat curse the system). So I decided to use Sophia, not any machine that Waggle people manage.

In Sophia, I was not about to clone a git repo because there is no proxy (`could not resolve proxy`), so:
```
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
```

and cloned jax ```https://github.com/jax-ml/jax#```
