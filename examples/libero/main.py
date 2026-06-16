import collections
import dataclasses
import json
import logging
import math
import pathlib
from datetime import datetime
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    record_dir: Optional[str] = None
    video_out_path: str = "data/libero/videos"

    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.record_dir is not None:
        root_dir = pathlib.Path(args.record_dir)
        video_out_dir = root_dir / "videos"
    else:
        video_out_dir = pathlib.Path(args.video_out_path)
        root_dir = video_out_dir.parent

    output_dir = root_dir / "output"

    root_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = output_dir / "episodes.json"
    run_summary_path = output_dir / "run_summary.json"
    task_summary_path = output_dir / "task_summary.json"
    metadata_path = output_dir / "metadata.json"

    metadata = {
        "created_at": datetime.now().isoformat(),
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "record_dir": str(root_dir),
        "output_dir": str(output_dir),
        "videos_dir": str(video_out_dir),
    }
    _write_json(metadata_path, metadata)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    episode_index = []
    task_summaries = []

    global_record_step = 0
    global_episode_num = 0

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0
        task_policy_calls = []

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []

            episode_start_idx = global_record_step

            logging.info(f"Starting episode {episode_idx + 1}...")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        global_record_step += 1

                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )

                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            episode_end_idx = global_record_step - 1
            num_policy_calls = max(0, episode_end_idx - episode_start_idx + 1)
            task_policy_calls.append(num_policy_calls)

            episode_index.append(
                {
                    "global_episode_num": int(global_episode_num),
                    "episode_num": int(episode_idx),
                    "task_id": int(task_id),
                    "task": str(task_description),
                    "start_idx": int(episode_start_idx),
                    "end_idx": int(episode_end_idx),
                    "success": bool(done),
                    "num_policy_calls": int(num_policy_calls),
                    "num_env_steps": int(t),
                }
            )

            global_episode_num += 1
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = _safe_filename(str(task_description))

            imageio.mimwrite(
                video_out_dir / f"rollout_task{task_id}_episode{episode_idx}_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            _write_json(episodes_path, episode_index)
            _write_json(
                run_summary_path,
                _make_run_summary(
                    args=args,
                    root_dir=root_dir,
                    output_dir=output_dir,
                    video_out_dir=video_out_dir,
                    num_tasks_in_suite=num_tasks_in_suite,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    global_record_step=global_record_step,
                ),
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"Wrote episode index to {episodes_path}")

        task_summary = {
            "task_id": int(task_id),
            "task": str(task_description),
            "episodes": int(task_episodes),
            "successes": int(task_successes),
            "success_rate": float(task_successes / task_episodes) if task_episodes else 0.0,
            "mean_policy_calls": float(np.mean(task_policy_calls)) if task_policy_calls else 0.0,
            "min_policy_calls": int(np.min(task_policy_calls)) if task_policy_calls else 0,
            "max_policy_calls": int(np.max(task_policy_calls)) if task_policy_calls else 0,
        }
        task_summaries.append(task_summary)
        _write_json(task_summary_path, task_summaries)

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    _write_json(episodes_path, episode_index)
    _write_json(task_summary_path, task_summaries)
    _write_json(
        run_summary_path,
        _make_run_summary(
            args=args,
            root_dir=root_dir,
            output_dir=output_dir,
            video_out_dir=video_out_dir,
            num_tasks_in_suite=num_tasks_in_suite,
            total_episodes=total_episodes,
            total_successes=total_successes,
            global_record_step=global_record_step,
        ),
    )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Wrote outputs to {output_dir}")


def _make_run_summary(
    *,
    args: Args,
    root_dir: pathlib.Path,
    output_dir: pathlib.Path,
    video_out_dir: pathlib.Path,
    num_tasks_in_suite: int,
    total_episodes: int,
    total_successes: int,
    global_record_step: int,
) -> dict:
    return {
        "task_suite_name": args.task_suite_name,
        "num_tasks": int(num_tasks_in_suite),
        "num_episodes": int(total_episodes),
        "num_successes": int(total_successes),
        "overall_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "num_policy_calls": int(global_record_step),
        "record_dir": str(root_dir),
        "output_dir": str(output_dir),
        "videos_dir": str(video_out_dir),
    }


def _write_json(path: pathlib.Path, data) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _safe_filename(text: str, max_len: int = 120) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:max_len] or "task"


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])

    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)