import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_BIPED_CFG, HUMANOID_LITE_LEG_JOINTS


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HUMANOID_LITE_LEG_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # Total Positive Weight: +4.0
    # Total Negative Weight: -14.662
    # Net Balance: −10.662

    # === Task Performance ===

    # Calculates the reward for how well linear/forward velocity aligns with intended.
    # Uses an exponential function.
    # If error is 0, output is +1. Else, less than 1.
    # Then we multiply by the weight (+2.0)
    # Category: Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=2.0,
    )
    
    # Calculates the reward for how well yaw rate (spinning) aligns with intended.
    # Uses an exponential function.
    # If error is 0, output is +1. Else, less than 1.
    # Then we multiply by the weight (+1.0)
    # Category: Task
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.0,
    )
    
    # We want the robot to take natural steps. 
    # This means rewarding how long the robot has one foot up and one foot down. 
    # We want at least 0.4s of air time. 
    # Category: Task
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"), # change from roll to pitch
            "threshold": 0.4,
        },
        weight=1.0,
    )

    # === Rewards for safety of the robot ===
    # Largest penalty, for when episode terminates early. 
    # This occurs when robot tilts more than 45 degrees
    # Weight is -10.0 to penalize early termination heavily
    # The robot must learn that falling is the worst possible outcome
    # Category: Safety
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )
    
    # We do not want the joints to surpass mechanical limits. Larger weight because this can break the joints
    # Category: Safety
    
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )
    
        # Penalize anything but the feet touching the ground. We want to avoid kneeling and crawling.
    # Category: Safety
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )
    # === Regulatory Rewards ===
    
    # Measures the squared z-axis velocity of the base (vertical movement). 
    # A smoothly walking robot's torso should stay at roughly constant height. 
    # Magnitude of weight is less than linear velocity reward term, because 
    # staying at the right speed is more important than staying at the right height.
    # Category: Regularization
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.1,
    )
    
    # Measures the squared pitch and roll of the base (x and y). 
    # A smoothly walking robot's torso should stay at roughly constant height. 
    # Magnitude of weight is less than linear velocity reward term, because 
    # staying at the right speed is more important than staying at the right height.
    # Category: Regularization
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    # Measures how tilted the robot currently is. It should stay upright.
    # This is the second largest weight in magnitude. Staying upright is 
    # important, because leaning will lead to falling.
    # Category: Safety 
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    # The action is the 10 joint angles. 
    # A large change in the action would look like jerky, sporatic movement. 
    # We want to keep the robot's movements as smooth as we can.
    # Category: Regularization
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    # Penalizes high joint torques that can damage actuator and 3D-printed hardware.
    # Gentle penalty, because we do not want the robot to avoid high torque if it is necessary.
    # Category: Regularization
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-2.0e-3,
    )
    
    # Penalizes high joint accelerations that can damage actuator and 3D-printed hardware.
    # Gentle penalty, because we do not want the robot to avoid high acceleration if it is necessary.
    # Category: Regularization
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-1.0e-6,
    )
    

    # Penalize feet sliding on the ground to exploit physics sim inaccuracies
    # Category: Regularization
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
        weight=-0.1,
    )

    # Penalize deviation from default of the joints that are not essential for locomotion.
    # Category: Regularization
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.2,
    )
    
    # DELETE THIS : no ankle roll joint in the 3ft HRVT URDF
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
        weight=-0.2,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.78, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class EventsCfg:
    """Configuration for events."""

    # === Startup behaviors ===
    # Helps sim2real by allowing the sim to account for slight variations in the material used
    # Randomizes static and dynamic friction of material
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        mode="startup",
    )
    # Randomizes mass districution parameters
    # Helps the sim account for variations in the mass of each component of the robot
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
        mode="startup",
    )
    # Randomizes starting positions of each joint
    # Helps simulation account for starting movement when joints are not aligned perfectly
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
        mode="startup",
    )
    # Randomizes stiffness and damping parameters
    # Helps with sim2real by letting simulation get used to movements with different actuators with different PD responses
    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        mode="startup",
    )

    # === Reset behaviors ===
    # Randomizes initial position/velocity values at the start of each episode
    # Helps with sim2real by gettign simulation used to different starting positions,
    # or starting with some initial motion, since in the real world it wont always start in a perfect position
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        mode="reset",
    )
    # Randomizes initial positions of the robot's joints
    # Helps with sim2real because the joints will not always be able to start at
    # the exact same position, so this gets the simulation used to that
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    # Randomizes the external torque applied to the robot
    # This helps with sim2real by allowing the robot to get used
    # to different environments/scenarios that could cause different
    # forces to be applied to it
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-2.0, 2.0),
            # "force_range": (-3.0, 3.0),
            # "torque_range": (-3.0, 3.0),
        },
        mode="reset",
    )

    # === Interval behaviors ===
    # Randomizes the interval between virtual shoves,
    # or alterations to the robot's velocity values
    # This helps sim2real because in a real environment
    # there are many things that could "push" the robot
    # so this adds that into the simulation
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    # )

@configclass
class CurriculumsCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class BerkeleyHumanoidLiteBipedEnvCfg(LocomotionVelocityEnvCfg):

    # Policy commands
    commands: CommandsCfg = CommandsCfg()

    # Policy observations
    observations: ObservationsCfg = ObservationsCfg()

    # Policy actions
    actions: ActionsCfg = ActionsCfg()

    # Policy rewards
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()

    # Randomization events
    events: EventsCfg = EventsCfg()

    # Curriculums
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Physics settings
        # 25 Hz override
        self.decimation = 8

        # Scene
        self.scene.robot = HUMANOID_LITE_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
