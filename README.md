# Project: Deep RL Quadcopter Controller
## Content: Deep Reinforcement Learning

### Description

Welcome to the Deep Reinforcement Learning project! In this project, we will demostrate how to build a pipeline to process a deep reinforcement agent which task or goal will be to learn how to flight a quadcopter. We would implement a Deep Deterministic Policy Gradients or DDPG algorithm. (Actor-Critic Method) 

The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own.

### Install

Follow instructions to create enviroment:

```
requirements/installation.txt
```
You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

### Code

Template code was provided by Udacity in the `Quadcopter_Project.ipynb` notebook file. It is also required to use the included python files in the `agents/` folder to run the app. While some code was already implemented to get me started, I needed to implement additional functionality when requested to successfully complete the project.

### Run

In a terminal or command window, navigate to the top-level project directory `Dog-Breed` (that contains this README) and run one of the following commands:

```bash
ipython notebook Quadcopter_Project.ipynb
```  
or
```bash
jupyter notebook Quadcopter_Project.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Authors

* **Udacity, Inc.** - *Udacity Instructor* -
* **Carlos Mertens** - *Udacity Student* -

## Acknowledgments

* Udacity, Inc.
