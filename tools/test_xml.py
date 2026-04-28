from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco


DEFAULT_XML = Path("assets/urdf/ur5_inspire_object.xml")


def print_model_summary(model: mujoco.MjModel) -> None:
    print("Model loaded")
    print(f"  bodies:    {model.nbody}")
    print(f"  joints:    {model.njnt}")
    print(f"  qpos/nv:   {model.nq}/{model.nv}")
    print(f"  actuators: {model.nu}")
    print()

    print("Joints")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qadr = model.jnt_qposadr[i]
        dadr = model.jnt_dofadr[i]
        low, high = model.jnt_range[i]
        print(f"  {i:02d} {name:28s} qpos[{qadr}] dof[{dadr}] range=({low:.4f}, {high:.4f})")

    print()
    print("Key bodies")
    for name in ("base_link", "shoulder_link", "wrist_3_link", "ee_link", "hand_base_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"  {name:18s}: {body_id}")


ROBOT_NU = 18
READY_ARM_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.2,
    "elbow_joint": 1.5,
    "wrist_1_joint": -0.4,
    "wrist_2_joint": 1.2,
    "wrist_3_joint": 0.0,
}

READY_HAND_QPOS = [
    0.35, 0.2, 0.15, 0.6,
    0.45, 0.8,
    0.45, 0.8,
    0.45, 0.8,
    0.45, 0.8,
]


def seed_ready_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_name, value in READY_ARM_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def sync_position_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # The first actuators in these test XMLs are position actuators for the robot.
    # Keeping ctrl equal to qpos makes viewer joint dragging feel like "hold here".
    count = min(model.nu, ROBOT_NU, model.nq)
    if count:
        data.ctrl[:count] = data.qpos[:count]
    if model.nu > ROBOT_NU:
        data.ctrl[ROBOT_NU:] = 0.0


def set_ready_controls(model: mujoco.MjModel, data: mujoco.MjData, close_hand: bool) -> None:
    for joint_name, value in READY_ARM_POSE.items():
        actuator_name = f"{joint_name}_pos"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if aid >= 0:
            data.ctrl[aid] = value
    if close_hand:
        for i, value in enumerate(READY_HAND_QPOS):
            aid = 6 + i
            if aid < model.nu:
                data.ctrl[aid] = value


def set_demo_controls(model: mujoco.MjModel, data: mujoco.MjData, close_hand: bool) -> None:
    set_ready_controls(model, data, close_hand)
    t = data.time
    # Small, slow oscillations around the ready pose. This proves the actuators are live
    # without violently throwing the fixed-base arm around.
    offsets = [
        0.45 * math.sin(0.8 * t),
        0.25 * math.sin(0.7 * t + 0.5),
        0.25 * math.sin(0.9 * t + 1.0),
        0.25 * math.sin(1.1 * t),
        0.20 * math.sin(1.0 * t + 1.2),
        0.35 * math.sin(1.3 * t),
    ]
    for i, offset in enumerate(offsets):
        if i < model.nu:
            data.ctrl[i] += offset


def run_headless(
    model: mujoco.MjModel,
    steps: int,
    ready_pose: bool,
    hold_current: bool,
    target_ready: bool,
    demo_arm: bool,
    close_hand: bool,
) -> None:
    data = mujoco.MjData(model)
    if ready_pose:
        seed_ready_pose(model, data)
    for _ in range(steps):
        if demo_arm:
            set_demo_controls(model, data, close_hand)
        elif target_ready:
            set_ready_controls(model, data, close_hand)
        elif hold_current:
            sync_position_controls(model, data)
        mujoco.mj_step(model, data)
    print(f"Headless step test passed: {steps} steps")


def run_viewer(
    model: mujoco.MjModel,
    ready_pose: bool,
    hold_current: bool,
    target_ready: bool,
    demo_arm: bool,
    close_hand: bool,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    if ready_pose:
        seed_ready_pose(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer opened. Close the MuJoCo window to exit.")
        if hold_current:
            print("Hold-current mode: position actuator targets follow qpos.")
        if target_ready:
            print("Target-ready mode: position actuator targets are set to a stable UR5 pose.")
        if demo_arm:
            print("Demo-arm mode: arm actuator targets move sinusoidally.")
        while viewer.is_running():
            if demo_arm:
                set_demo_controls(model, data, close_hand)
            elif target_ready:
                set_ready_controls(model, data, close_hand)
            elif hold_current:
                sync_position_controls(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ready-pose", action="store_true")
    parser.add_argument("--hold-current", action="store_true")
    parser.add_argument("--target-ready", action="store_true")
    parser.add_argument("--demo-arm", action="store_true")
    parser.add_argument("--close-hand", action="store_true")
    args = parser.parse_args()

    if args.viewer and not any((args.hold_current, args.target_ready, args.demo_arm)):
        args.ready_pose = True
        args.target_ready = True
        print("Viewer default: enabling --ready-pose --target-ready for a stable display.")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    print_model_summary(model)

    if args.viewer:
        run_viewer(
            model,
            args.ready_pose,
            args.hold_current,
            args.target_ready,
            args.demo_arm,
            args.close_hand,
        )
    else:
        run_headless(
            model,
            args.steps,
            args.ready_pose,
            args.hold_current,
            args.target_ready,
            args.demo_arm,
            args.close_hand,
        )


if __name__ == "__main__":
    main()
