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

    # === Reward for task-space performance ===
    
    # Calculates how well the robot conforms to the commanded linear/forward velocity.
    # Returns +1.0 for perfect tracking (Error = 0)
    # Category: Task
    # Priority: High (Weight = +2.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=2.0,
    )

    # Calculates how well the robot conforms to the commanded yaw/spin velocity.
    # Returns +1.0 for perfect tracking (Error = 0)
    # Category: Task
    # Priority: Medium (Weight = +1.0)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.0,
    )

    # === Reward for basic behaviors ===
    # A massive penalty for when the robot terminates an episode early (falling)
    # Administers a -10.0 penalty, which allows the robot to learn this is the
    # worst scenario and needs to be avoided.
    # Category: Safety/constraint
    # Priority: Highest (Weight = -10.0)
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # Measures the velocity on the z-axis (vertical bouncing/movement)
    # Administers a -0.1 penalty to encourage motion at constant height
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -1.0)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.1,
    )

    # Measures the angular velocity on the xy-plane (wobbling)
    # Applies a -0.05 penalty to encourage non-wobbly motion
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.05)
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # Measures the tilt of the robot's body. Applies a strong -2.0 penalty
    # to ensure the robot stays upright (centers mass, helps prevent falling)
    # Category: Safety/constraint
    # Priority: High (Weight = -2.0)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    # Penalizes rapid changes in the robot's actions
    # Helps minimize jitteriness, encourages smoother movements
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.01)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Penalizes high torques at the joints in order to encourage
    # energy efficiency and protect hardware/3D-printed material.
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.002)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-2.0e-3,
    )

    # Penalizes high acceleration at the joints in order to reduce
    # stress on hardware/3D-printed gears and encourage smoother joint motion.
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -1e-6)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-1.0e-6,
    )

    # Penalizes when the joints are near the position limits
    # This encourages the robot to stay away from mechanical stops.
    # Category: Safety/constraint
    # Priority: Medium (Weight = -1.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    # === Reward for encouraging behaviors ===
    # Rewards the robot for having a natural stance when walking (1 leg up, 1 leg down)
    # Requires one leg to be in air for at least 0.4s. Inactive when stationary.
    # Category: Task
    # Priority: Medium (Weight = +1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.4,
        },
        weight=1.0,
    )
    
    # Penalizes the robot for sliding feet while touching the ground
    # This prevents abuse of simulation exploits, and prevents soles from scraping
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.1)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
        weight=-0.1,
    )

    # Penalizes the robot for touching the ground with any limbs except feet
    # Category: Safety/constraint
    # Priority: Medium (Weight = -1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )

    # Penalizes the robot when hip joints deviate from default positioning
    # This helps keep the legs aligned and pointing forward under the body.
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.2)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.2,
    )

    # Penalizes the robot when ankle joints deviate from default positioning
    # NOTE: This is NOT a joint that is present on the 3ft URDF robot model
    # Category: Regularization
    # Priority: Fine-Tuning (Weight = -0.2)
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
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
        mode="startup",
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
        mode="startup",
    )
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
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
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
