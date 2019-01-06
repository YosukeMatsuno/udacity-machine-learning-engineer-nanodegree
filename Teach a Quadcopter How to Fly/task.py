import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat *6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        #for reward function
        self.z_bonus = 5.0
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        z_diff =  self.sim.pose[2]- self.target_pos[2] #negative means current lower than target
        z_factor = self.z_bonus if z_diff >= 0 else 5.0 #z_bonus is 5.0

        # The closer the better the reward
        if z_diff<=0: #if current is below target
            distance = abs(z_diff) 
            reward += (1 / distance) * z_factor
        else: #if higher than target
            distance = z_diff
            reward += distance * z_factor
            
#         penalty = abs(abs(current_pos - self.target_pos).sum() - abs(self.sim.v).sum())
#         reward -= penalty
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state