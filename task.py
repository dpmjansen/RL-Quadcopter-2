import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, debug=False):
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

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.debug = debug

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        pose_reward = 10.-3.*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        equality_reward = 1 - np.average(self.sim.prop_wind_speed)
        body_velocity_reward_arr = np.absolute(self.sim.find_body_velocity()) # lower velocities should give higher reward
        body_velocity_reward = 0
        for x in np.nditer(body_velocity_reward_arr):
            body_velocity_reward += 1 - x
            
        reward = pose_reward + equality_reward + body_velocity_reward
        
        if self.debug:
            print("--- get_reward function ---")            
            print("pose_reward:" + str(pose_reward))
            print("equality_reward:" + str(equality_reward))
            print("body_velocity_reward:" + str(body_velocity_reward_arr))
            print("body_velocity_reward:" + str(body_velocity_reward))
            print("reward:" + str(reward))
            print("---")
            
#        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
#        sum_acceleration = np.linalg.norm(self.sim.linear_accel)
#        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05
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
        
        if self.debug:
            print("--- step function ---")
            print("next_state: " + str(next_state))
            print("reward: " + str(reward))
            print("done: " + str(done))
            print("---")
            
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state