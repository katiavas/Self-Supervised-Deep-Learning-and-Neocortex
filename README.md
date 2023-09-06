# Bachelor Thesis - Reinforcement Learning : Self-Supervised Deep Learning and Neocortex

# Abstract 
The exploration exploitation dilemma is one of the most prominent open research problems in Reinforcement Learning (RL). Pathak et al [67] have proposed a solution that introduces balance between those
two components. The solution involves an agent, that is the RL algorithm , producing its own intrinsic
rewards, through a bolt-on module, namely Intrinsic Curiosity Module(ICM). Using intrinsic rewards
the algorithm is guided towards more exploratory behaviour but still not loosing sight of its ultimate
goal of maximising its learning progress, which is quantified by the extrinsic rewards it has accumulated
by the end of the training. The first aim of this project is to implement this solution in environments
with dense discrete extrinsic rewards supplied to the agent and importantly modify the implementation
so that it works in Atari environments. The results of the implementation align with the one;s of the
original creator of the ICM, as it is deduced that intrinsic curiosity boosts the learning progress of the
agent. Moreover, as it has been suggested by prior research a form of intrinsic reward is being produced
in the brain, which is in this research hypothesis it is proposed that it is processed in the same way as the
ICM. The ICM is then paralleled with the functionality of the neocortex which is believed to be the main
component of the brain that drives learning and is responsible for functionalities that distinguish humans
from other mammals. The results of this implementation are in a promising direction as it is illustrated
that the more biologically plausible model does yield even better results than the original ICM. However,
this must be subject of further research as the reasons underlying the boost in learning performance of
the more biologically plausible ICM compared to the original ICM remain in debate

# Supporting Technologies
<pre>
•  Python as a programming language to implement the ICM.</pre>
<pre>

• PyTorch Python’s Machine Learning Framework to implement the ICM, this
include using convolutional neural networks for encoding features of incoming images
and neural networks for making predictions.</pre>
<pre>

• OpenCV computer vision library to process incoming raw input
images so that computations are more efficient.</pre>
<pre>
• Open-AI Gym open source toolkit containing a range of environments where
the agent can learn, but also allowing to directly modify the open-AI Gym environment
by using Gym wrappers.
</pre>

<pre>
  • Using Google's DeepMind A3C agent as a baseline </pre>
