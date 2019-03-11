import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.
    
    The task for this environment is to initiate a zero position and to takeoff to as smooth as possible.
    The goal is to reach position x=0, y=0 and Z(high)=10.
    """
    
    
    def __init__(self, runtime=100., target_pos=np.array([0., 0., 50.])):
        """Initialize a Task object.
        
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        
        # The quadcopter's starting position
        self.init_pose = np.array([0., 0., 0., 0., 0., 0.])
        self.init_velocities = np.array([0., 0., 0.])
        self.init_angle_velocities = np.array([0., 0., 0.])
        self.target_pos = target_pos
        
        # Simulation
        self.sim = PhysicsSim(self.init_pose, self.init_velocities, self.init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.accumulative_rewards = []


    def get_reward(self):
        """Uses current pose of sim to return reward.
        
        The agent will get rewarded according to best approximation to the each
        coordinate x, y, z.
        """
        reward_x = 1.-.003*(abs(self.sim.pose[0] - self.target_pos[0]))
        reward_y = 1.-.003*(abs(self.sim.pose[1] - self.target_pos[1]))
        reward_z = 1.-.003*(abs(self.sim.pose[2] - self.target_pos[2]))
        reward = reward_x + reward_y + reward_z
        
        # TODO: To ensure the stability of the agent, try using reward clipping.
        #  Reference: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.clip.html
        
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        self.accumulative_rewards.append(reward)  # to accumulate reward to be plot
        #print(reward)
        
        return next_state, reward, done

    
    def reset(self):
        """Reset the sim to start a new episode."""
        
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
