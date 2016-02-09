# Deep Reinforcement Learning Papers
A list of recent papers regarding deep reinforcement learning. <br>
The papers are organized based on manually-defined bookmarks. <br>
They are sorted by time to see the recent papers first. <br>
Any suggestions and pull requests are welcome. 

# Bookmarks
  * [All Papers](#all-papers)
  * [Value Function Approximation](#value-function-approximation)
  * [Policy Gradient](#policy-gradient)
  * [Discrete Control](#discrete-control)
  * [Continuous Control](#continuous-control)
  * [Text Domain](#text-domain)
  * [Visual Domain](#visual-domain)
  * [Robotics](#robotics)
  * [Games](#games)
  * [Monte-Carlo Tree Search](#monte-carlo-tree-search)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
  * [Improving Exploration](#improving-exploration)
  * [Transfer Learning](#transfer-learning)
  * [Multi-Agent](#multi-agent)

## All Papers
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Memory-based control with recurrent neural networks](http://arxiv.org/abs/1512.04455), N. Heess et al., *NIPS Workshop*, 2015.
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
  * [Strategic Dialogue Management via Deep Reinforcement Learning](http://arxiv.org/abs/1511.08099), H. Cuayáhuitl et al., *NIPS Workshop*, 2015.
  * [MazeBase: A Sandbox for Learning from Games](http://arxiv.org/abs/1511.07401), S. Sukhbaatar et al., *arXiv*, 2016.
  * [Learning Simple Algorithms from Examples](http://arxiv.org/abs/1511.07275), W. Zaremba et al., *arXiv*, 2015.
  * [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Z. Wang et al., *arXiv*, 2015.
  * [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342), E. Parisotto, et al., *ICLR*, 2016.
  * [Better Computer Go Player with Neural Network and Long-term Prediction](http://arxiv.org/abs/1511.06410), Y. Tian et al., *ICLR*, 2016.
  * [Policy Distillation](http://arxiv.org/abs/1511.06295), A. A. Rusu et at., *ICLR*, 2016.
  * [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), T. Schaul et al., *ICLR*, 2016.
  * [Deep Reinforcement Learning with an Action Space Defined by Natural Language](http://arxiv.org/abs/1511.04636), J. He et al., *arXiv*, 2015.
  * [Deep Reinforcement Learning in Parameterized Action Space](http://arxiv.org/abs/1511.04143), M. Hausknecht et al., *ICLR*, 2016.
  * [Towards Vision-Based Deep Reinforcement Learning for Robotic Motion Control](http://arxiv.org/abs/1511.03791), F. Zhang et al., *arXiv*, 2015.
  * [Generating Text with Deep Reinforcement Learning](http://arxiv.org/abs/1510.09202), H. Guo, *arXiv*, 2015.
  * [ADAAPT: A Deep Architecture for Adaptive Policy Transfer from Multiple Sources](http://arxiv.org/abs/1510.02879), J. Rajendran et al., *arXiv*, 2015. 
  * [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](http://arxiv.org/abs/1509.08731), S. Mohamed and D. J. Rezende, *arXiv*, 2015.
  * [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), H. van Hasselt et al., *arXiv*, 2015.
  * [Recurrent Reinforcement Learning: A Hybrid Approach](http://arxiv.org/abs/1509.03044), X. Li et al., *arXiv*, 2015. 
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf), K. Narasimhan et al., *EMNLP*, 2015.
  * [Giraffe: Using Deep Reinforcement Learning to Play Chess](http://arxiv.org/abs/1509.01549), M. Lai, *arXiv*, 2015.
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Learning Continuous Control Policies by Stochastic Value Gradients](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf), N. Heess et al., *NIPS*, 2015.
  * [Learning Deep Neural Network Policies with Continuous Memory States](http://arxiv.org/abs/1507.01273), M. Zhang et al., *arXiv*, 2015.
  * [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527), M. Hausknecht and P. Stone, *arXiv*, 2015.
  * [Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to Action Sequences](http://arxiv.org/abs/1506.04089), H. Mei et al., *arXiv*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.
  * [Maximum Entropy Deep Inverse Reinforcement Learning](http://arxiv.org/abs/1507.04888), M. Wulfmeier et al., *arXiv*, 2015.
  * [High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438), J. Schulman et al., *ICLR*, 2016.
  * [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702), S. Levine et al., *arXiv*, 2015.
  * [DeepMPC: Learning Deep Latent Features for
Model Predictive Control](http://deepmpc.cs.cornell.edu/DeepMPC.pdf), I. Lenz, et al., *RSS*, 2015.
  * [Universal Value Function Approximators](http://schaul.site44.com/publications/uvfa.pdf), T. Schaul et al., *ICML*, 2015.
  * [Deterministic Policy Gradient Algorithms](http://jmlr.org/proceedings/papers/v32/silver14.pdf), D. Silver et al., *ICML*, 2015.
  * [Massively Parallel Methods for Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/gorila.pdf), A. Nair et al., *ICML Workshop*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.
  * [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.
  * [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), X. Guo et al., *NIPS*, 2014.
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.

## Value Function Approximation
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
  * [Strategic Dialogue Management via Deep Reinforcement Learning](http://arxiv.org/abs/1511.08099), H. Cuayáhuitl et al., *NIPS Workshop*, 2015.
  * [Learning Simple Algorithms from Examples](http://arxiv.org/abs/1511.07275), W. Zaremba et al., *arXiv*, 2015.
  * [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Z. Wang et al., *arXiv*, 2015.
  * [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), T. Schaul et al., *ICLR*, 2016.
  * [Deep Reinforcement Learning with an Action Space Defined by Natural Language](http://arxiv.org/abs/1511.04636), J. He et al., *arXiv*, 2015.
  * [Deep Reinforcement Learning in Parameterized Action Space](http://arxiv.org/abs/1511.04143), M. Hausknecht et al., *ICLR*, 2016.
  * [Towards Vision-Based Deep Reinforcement Learning for Robotic Motion Control](http://arxiv.org/abs/1511.03791), F. Zhang et al., *arXiv*, 2015.
  * [Generating Text with Deep Reinforcement Learning](http://arxiv.org/abs/1510.09202), H. Guo, *arXiv*, 2015.
  * [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), H. van Hasselt et al., *arXiv*, 2015.
  * [Recurrent Reinforcement Learning: A Hybrid Approach](http://arxiv.org/abs/1509.03044), X. Li et al., *arXiv*, 2015. 
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf), K. Narasimhan et al., *EMNLP*, 2015.
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527), M. Hausknecht and P. Stone, *arXiv*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.
  * [Massively Parallel Methods for Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/gorila.pdf), A. Nair et al., *ICML Workshop*, 2015.
  * [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.

## Policy Gradient
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Memory-based control with recurrent neural networks](http://arxiv.org/abs/1512.04455), N. Heess et al., *NIPS Workshop*, 2015.
  * [MazeBase: A Sandbox for Learning from Games](http://arxiv.org/abs/1511.07401), S. Sukhbaatar et al., *arXiv*, 2016.
  * [ADAAPT: A Deep Architecture for Adaptive Policy Transfer from Multiple Sources](http://arxiv.org/abs/1510.02879), J. Rajendran et al., *arXiv*, 2015.
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Learning Continuous Control Policies by Stochastic Value Gradients](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf), N. Heess et al., *NIPS*, 2015.
  * [High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438), J. Schulman et al., *ICLR*, 2016.
  * [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702), S. Levine et al., *arXiv*, 2015.
  * [Deterministic Policy Gradient Algorithms](http://jmlr.org/proceedings/papers/v32/silver14.pdf), D. Silver et al., *ICML*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.

## Discrete Control
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
  * [Strategic Dialogue Management via Deep Reinforcement Learning](http://arxiv.org/abs/1511.08099), H. Cuayáhuitl et al., *NIPS Workshop*, 2015.
  * [Learning Simple Algorithms from Examples](http://arxiv.org/abs/1511.07275), W. Zaremba et al., *arXiv*, 2015.
  * [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Z. Wang et al., *arXiv*, 2015.
  * [Better Computer Go Player with Neural Network and Long-term Prediction](http://arxiv.org/abs/1511.06410), Y. Tian et al., *ICLR*, 2016.
  * [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342), E. Parisotto, et al., *ICLR*, 2016.
  * [Policy Distillation](http://arxiv.org/abs/1511.06295), A. A. Rusu et at., *ICLR*, 2016.
  * [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), T. Schaul et al., *ICLR*, 2016.
  * [Deep Reinforcement Learning with an Action Space Defined by Natural Language](http://arxiv.org/abs/1511.04636), J. He et al., *arXiv*, 2015.
  * [Deep Reinforcement Learning in Parameterized Action Space](http://arxiv.org/abs/1511.04143), M. Hausknecht et al., *ICLR*, 2016.
  * [Towards Vision-Based Deep Reinforcement Learning for Robotic Motion Control](http://arxiv.org/abs/1511.03791), F. Zhang et al., *arXiv*, 2015.
  * [Generating Text with Deep Reinforcement Learning](http://arxiv.org/abs/1510.09202), H. Guo, *arXiv*, 2015.
  * [ADAAPT: A Deep Architecture for Adaptive Policy Transfer from Multiple Sources](http://arxiv.org/abs/1510.02879), J. Rajendran et al., *arXiv*, 2015.
  * [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](http://arxiv.org/abs/1509.08731), S. Mohamed and D. J. Rezende, *arXiv*, 2015.
  * [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), H. van Hasselt et al., *arXiv*, 2015.
  * [Recurrent Reinforcement Learning: A Hybrid Approach](http://arxiv.org/abs/1509.03044), X. Li et al., *arXiv*, 2015.
  * [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf), K. Narasimhan et al., *EMNLP*, 2015.
  * [Giraffe: Using Deep Reinforcement Learning to Play Chess](http://arxiv.org/abs/1509.01549), M. Lai, *arXiv*, 2015.
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527), M. Hausknecht and P. Stone, *arXiv*, 2015.
  * [Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to Action Sequences](http://arxiv.org/abs/1506.04089), H. Mei et al., *arXiv*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.
  * [Universal Value Function Approximators](http://schaul.site44.com/publications/uvfa.pdf), T. Schaul et al., *ICML*, 2015.
  * [Massively Parallel Methods for Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/gorila.pdf), A. Nair et al., *ICML Workshop*, 2015.
  * [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.
  * [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), X. Guo et al., *NIPS*, 2014.
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.

## Continuous Control
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Memory-based control with recurrent neural networks](http://arxiv.org/abs/1512.04455), N. Heess et al., *NIPS Workshop*, 2015.
  * [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](http://arxiv.org/abs/1509.08731), S. Mohamed and D. J. Rezende, *arXiv*, 2015.
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Learning Continuous Control Policies by Stochastic Value Gradients](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf), N. Heess et al., *NIPS*, 2015.
  * [Learning Deep Neural Network Policies with Continuous Memory States](http://arxiv.org/abs/1507.01273), M. Zhang et al., *arXiv*, 2015.
  * [High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438), J. Schulman et al., *ICLR*, 2016.
  * [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702), S. Levine et al., *arXiv*, 2015.
  * [DeepMPC: Learning Deep Latent Features for
Model Predictive Control](http://deepmpc.cs.cornell.edu/DeepMPC.pdf), I. Lenz, et al., *RSS*, 2015.
  * [Deterministic Policy Gradient Algorithms](http://jmlr.org/proceedings/papers/v32/silver14.pdf), D. Silver et al., *ICML*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.

## Text Domain
  * [Strategic Dialogue Management via Deep Reinforcement Learning](http://arxiv.org/abs/1511.08099), H. Cuayáhuitl et al., *NIPS Workshop*, 2015.
  * [MazeBase: A Sandbox for Learning from Games](http://arxiv.org/abs/1511.07401), S. Sukhbaatar et al., *arXiv*, 2016.
  * [Deep Reinforcement Learning with an Action Space Defined by Natural Language](http://arxiv.org/abs/1511.04636), J. He et al., *arXiv*, 2015.
  * [Generating Text with Deep Reinforcement Learning](http://arxiv.org/abs/1510.09202), H. Guo, *arXiv*, 2015.
  * [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf), K. Narasimhan et al., *EMNLP*, 2015.
  * [Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to Action Sequences](http://arxiv.org/abs/1506.04089), H. Mei et al., *arXiv*, 2015.

## Visual Domain
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Memory-based control with recurrent neural networks](http://arxiv.org/abs/1512.04455), N. Heess et al., *NIPS Workshop*, 2015.
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
  * [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Z. Wang et al., *arXiv*, 2015.
  * [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342), E. Parisotto, et al., *ICLR*, 2016.
  * [Better Computer Go Player with Neural Network and Long-term Prediction](http://arxiv.org/abs/1511.06410), Y. Tian et al., *ICLR*, 2016.
  * [Policy Distillation](http://arxiv.org/abs/1511.06295), A. A. Rusu et at., *ICLR*, 2016.
  * [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), T. Schaul et al., *ICLR*, 2016.
  * [Deep Reinforcement Learning in Parameterized Action Space](http://arxiv.org/abs/1511.04143), M. Hausknecht et al., *ICLR*, 2016.
  * [Towards Vision-Based Deep Reinforcement Learning for Robotic Motion Control](http://arxiv.org/abs/1511.03791), F. Zhang et al., *arXiv*, 2015.
  * [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](http://arxiv.org/abs/1509.08731), S. Mohamed and D. J. Rezende, *arXiv*, 2015.
  * [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), H. van Hasselt et al., *arXiv*, 2015.
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Giraffe: Using Deep Reinforcement Learning to Play Chess](http://arxiv.org/abs/1509.01549), M. Lai, *arXiv*, 2015.
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Learning Continuous Control Policies by Stochastic Value Gradients](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf), N. Heess et al., *NIPS*, 2015.
  * [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527), M. Hausknecht and P. Stone, *arXiv*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.
  * [High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438), J. Schulman et al., *ICLR*, 2016.
  * [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702), S. Levine et al., *arXiv*, 2015.
  * [Universal Value Function Approximators](http://schaul.site44.com/publications/uvfa.pdf), T. Schaul et al., *ICML*, 2015.
  * [Massively Parallel Methods for Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/gorila.pdf), A. Nair et al., *ICML Workshop*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.
  * [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.
  * [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), X. Guo et al., *NIPS*, 2014.
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.

## Robotics
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Memory-based control with recurrent neural networks](http://arxiv.org/abs/1512.04455), N. Heess et al., *NIPS Workshop*, 2015.
  * [Towards Vision-Based Deep Reinforcement Learning for Robotic Motion Control](http://arxiv.org/abs/1511.03791), F. Zhang et al., *arXiv*, 2015.
  * [Learning Continuous Control Policies by Stochastic Value Gradients](http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients.pdf), N. Heess et al., *NIPS*, 2015.
  * [Learning Deep Neural Network Policies with Continuous Memory States](http://arxiv.org/abs/1507.01273), M. Zhang et al., *arXiv*, 2015.
  * [High-Dimensional Continuous Control Using Generalized Advantage Estimation](http://arxiv.org/abs/1506.02438), J. Schulman et al., *ICLR*, 2016.
  * [End-to-End Training of Deep Visuomotor Policies](http://arxiv.org/abs/1504.00702), S. Levine et al., *arXiv*, 2015.
  * [DeepMPC: Learning Deep Latent Features for
Model Predictive Control](http://deepmpc.cs.cornell.edu/DeepMPC.pdf), I. Lenz, et al., *RSS*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.

## Games
  * [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783), V. Mnih et al., *arXiv*, 2016.
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
  * [MazeBase: A Sandbox for Learning from Games](http://arxiv.org/abs/1511.07401), S. Sukhbaatar et al., *arXiv*, 2016.
  * [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581), Z. Wang et al., *arXiv*, 2015.
  * [Better Computer Go Player with Neural Network and Long-term Prediction](http://arxiv.org/abs/1511.06410), Y. Tian et al., *ICLR*, 2016.
  * [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342), E. Parisotto, et al., *ICLR*, 2016.
  * [Policy Distillation](http://arxiv.org/abs/1511.06295), A. A. Rusu et at., *ICLR*, 2016.
  * [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952), T. Schaul et al., *ICLR*, 2016.
  * [Deep Reinforcement Learning with an Action Space Defined by Natural Language](http://arxiv.org/abs/1511.04636), J. He et al., *arXiv*, 2015.
  * [Deep Reinforcement Learning in Parameterized Action Space](http://arxiv.org/abs/1511.04143), M. Hausknecht et al., *ICLR*, 2016.
  * [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](http://arxiv.org/abs/1509.08731), S. Mohamed and D. J. Rezende, *arXiv*, 2015.
  * [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461), H. van Hasselt et al., *arXiv*, 2015.
  * [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971), T. P. Lillicrap et al., *ICLR*, 2016.
  * [Language Understanding for Text-based Games Using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf), K. Narasimhan et al., *EMNLP*, 2015.
  * [Giraffe: Using Deep Reinforcement Learning to Play Chess](http://arxiv.org/abs/1509.01549), M. Lai, *arXiv*, 2015.
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527), M. Hausknecht and P. Stone, *arXiv*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.
  * [Universal Value Function Approximators](http://schaul.site44.com/publications/uvfa.pdf), T. Schaul et al., *ICML*, 2015.
  * [Massively Parallel Methods for Deep Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/gorila.pdf), A. Nair et al., *ICML Workshop*, 2015.
  * [Trust Region Policy Optimization](http://jmlr.org/proceedings/papers/v37/schulman15.pdf), J. Schulman et al., *ICML*, 2015.
  * [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.
  * [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), X. Guo et al., *NIPS*, 2014.
  * [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), V. Mnih et al., *NIPS Workshop*, 2013.

## Monte-Carlo Tree Search
  * [Mastering the game of Go with deep neural networks and tree search](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), D. Silver et al., *Nature*, 2016.
  * [Better Computer Go Player with Neural Network and Long-term Prediction](http://arxiv.org/abs/1511.06410), Y. Tian et al., *ICLR*, 2016.
  * [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), X. Guo et al., *NIPS*, 2014.

## Inverse Reinforcement Learning
  * [Maximum Entropy Deep Inverse Reinforcement Learning](http://arxiv.org/abs/1507.04888), M. Wulfmeier et al., *arXiv*, 2015.

## Transfer Learning
  * [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](http://arxiv.org/abs/1511.06342), E. Parisotto, et al., *ICLR*, 2016.
  * [Policy Distillation](http://arxiv.org/abs/1511.06295), A. A. Rusu et at., *ICLR*, 2016.
  * [ADAAPT: A Deep Architecture for Adaptive Policy Transfer from Multiple Sources](http://arxiv.org/abs/1510.02879), J. Rajendran et al., *arXiv*, 2015.
  * [Universal Value Function Approximators](http://schaul.site44.com/publications/uvfa.pdf), T. Schaul et al., *ICML*, 2015.

## Improving Exploration
  * [Action-Conditional Video Prediction using Deep Networks in Atari Games](http://arxiv.org/abs/1507.08750), J. Oh et al., *NIPS*, 2015.
  * [Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](http://arxiv.org/abs/1507.00814), B. C. Stadie et al., *arXiv*, 2015.

## Multi Agent
  * [Multiagent Cooperation and Competition with Deep Reinforcement Learning](http://arxiv.org/abs/1511.08779), A. Tampuu et al., *arXiv*, 2015.
