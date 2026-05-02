#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RAW_REPO_ID = "zhuoKCL/prgvla"
CAMERA_KEY = "310222076420_left"
IMAGE_KEY = "observation.images.camera_left"
ROBOT_TYPE = "fr3"
JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]

TASK_SPECS = {
    "sorting": {
        "task_text": "sort the table",
        "dataset_name": "prgvla_sort_the_table_leftcam_v1",
    },
    "stack": {
        "task_text": "stack the cups",
        "dataset_name": "prgvla_stack_the_cups_leftcam_v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert zhuoKCL/prgvla episodes to LeRobot v3 datasets.")
    parser.add_argument(
        "--task",
        choices=["sorting", "stack", "all"],
        default="all",
        help="Which PRGVLA task split to convert.",
    )
    parser.add_argument(
        "--repo-id",
        default=RAW_REPO_ID,
        help="Raw Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/home/n84416302/dataset/raw/prgvla"),
        help="Where to mirror the raw PRGVLA files locally.",
    )
    parser.add_argument(
        "--output-dir-base",
        type=Path,
        default=Path("/home/n84416302/dataset"),
        help="Base directory for converted LeRobot datasets.",
    )
    parser.add_argument(
        "--camera-key",
        default=CAMERA_KEY,
        help="Raw PRGVLA camera file stem to use.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download and use files already present in --download-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the selected raw files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes per task, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing converted output directory before writing a new one.",
    )
    parser.add_argument(
        "--only-successful",
        action="store_true",
        default=True,
        help="Skip raw episodes marked with success=false in trajectory attrs.",
    )
    parser.add_argument(
        "--include-failed",
        dest="only_successful",
        action="store_false",
        help="Keep failed raw episodes instead of skipping them.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of LeRobot image writer threads.",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="Codec to use when LeRobot re-encodes the selected camera stream.",
    )
    return parser.parse_args()


def make_features(height: int, width: int) -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": JOINT_NAMES,
        },
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def download_task_split(repo_id: str, task_name: str, camera_key: str, download_dir: Path, force_download: bool) -> None:
    patterns = [
        f"{task_name}/*/trajectory.h5",
        f"{task_name}/*/recordings/{camera_key}.mp4",
    ]
    logging.info("Downloading raw split '%s' with patterns: %s", task_name, patterns)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=patterns,
        force_download=force_download,
    )


def get_episode_dirs(raw_task_root: Path, max_episodes: int | None) -> list[Path]:
    episodes = []
    for episode_dir in sorted(raw_task_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        if (episode_dir / "trajectory.h5").is_file():
            episodes.append(episode_dir)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def inspect_video(video_path: Path) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, size={width}x{height}")
    return fps, width, height, frame_count


def load_episode_arrays(h5_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    with h5py.File(h5_path, "r") as h5f:
        success = bool(h5f.attrs.get("success", True))
        obs_joint = np.asarray(h5f["observation/robot_state/joint_positions"][:], dtype=np.float32)
        obs_gripper = np.asarray(h5f["observation/robot_state/gripper_position"][:], dtype=np.float32).reshape(-1, 1)
        action_joint = np.asarray(h5f["action/joint_position"][:], dtype=np.float32)
        action_gripper = np.asarray(h5f["action/gripper_position"][:], dtype=np.float32).reshape(-1, 1)

    state = np.concatenate([obs_joint, obs_gripper], axis=1, dtype=np.float32)
    action = np.concatenate([action_joint, action_gripper], axis=1, dtype=np.float32)
    return state, action, success


def ensure_clean_output(output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Remove it manually or pass --overwrite."
        )
    shutil.rmtree(output_root)


def convert_single_task(task_name: str, args: argparse.Namespace) -> None:
    spec = TASK_SPECS[task_name]
    raw_task_root = args.download_dir / task_name
    if not raw_task_root.exists():
        raise FileNotFoundError(f"Raw task directory not found: {raw_task_root}")

    episode_dirs = get_episode_dirs(raw_task_root, args.max_episodes)
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found under {raw_task_root}")

    sample_video = episode_dirs[0] / "recordings" / f"{args.camera_key}.mp4"
    fps, width, height, _ = inspect_video(sample_video)

    dataset_name = spec["dataset_name"]
    output_root = args.output_dir_base / dataset_name
    ensure_clean_output(output_root, args.overwrite)

    logging.info(
        "Creating LeRobot dataset '%s' at %s from %d '%s' episodes using camera %s",
        dataset_name,
        output_root,
        len(episode_dirs),
        task_name,
        args.camera_key,
    )

    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        root=output_root,
        fps=fps,
        features=make_features(height, width),
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
        streaming_encoding=True,
    )

    converted_episodes = 0
    skipped_failed = 0
    total_frames = 0

    try:
        for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
            h5_path = episode_dir / "trajectory.h5"
            video_path = episode_dir / "recordings" / f"{args.camera_key}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video file: {video_path}")

            state, action, success = load_episode_arrays(h5_path)
            if args.only_successful and not success:
                skipped_failed += 1
                logging.info("Skipping failed episode %s", episode_dir.name)
                continue

            video_fps, video_width, video_height, video_frames = inspect_video(video_path)
            if (video_fps, video_width, video_height) != (fps, width, height):
                raise ValueError(
                    f"Video metadata mismatch for {video_path}: "
                    f"got fps={video_fps}, size={video_width}x{video_height}, "
                    f"expected fps={fps}, size={width}x{height}"
                )

            target_len = min(len(state), len(action), video_frames)
            if target_len <= 0:
                logging.warning("Skipping empty episode %s", episode_dir.name)
                continue
            if target_len != len(state) or target_len != len(action) or target_len != video_frames:
                logging.warning(
                    "Length mismatch in %s: state=%d action=%d video=%d. Truncating to %d frames.",
                    episode_dir.name,
                    len(state),
                    len(action),
                    video_frames,
                    target_len,
                )

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to read video: {video_path}")

            frames_written = 0
            try:
                for frame_idx in range(target_len):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        logging.warning(
                            "Video decode stopped early for %s at frame %d/%d",
                            video_path,
                            frame_idx,
                            target_len,
                        )
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(
                        {
                            "observation.state": state[frame_idx],
                            "action": action[frame_idx],
                            IMAGE_KEY: frame_rgb,
                            "task": spec["task_text"],
                        }
                    )
                    frames_written += 1
            except Exception:
                dataset.clear_episode_buffer(delete_images=True)
                raise
            finally:
                cap.release()

            if frames_written == 0:
                logging.warning("Skipping zero-frame episode %s", episode_dir.name)
                dataset.clear_episode_buffer(delete_images=True)
                continue

            if frames_written != target_len:
                logging.warning(
                    "Episode %s ended with %d/%d decoded frames. Saving the decoded prefix only.",
                    episode_dir.name,
                    frames_written,
                    target_len,
                )

            dataset.save_episode()
            converted_episodes += 1
            total_frames += frames_written
            logging.info(
                "[%d/%d] saved %s with %d frames",
                episode_idx,
                len(episode_dirs),
                episode_dir.name,
                frames_written,
            )
    finally:
        dataset.finalize()

    logging.info(
        "Finished %s -> %s | converted episodes=%d skipped_failed=%d total_frames=%d",
        task_name,
        output_root,
        converted_episodes,
        skipped_failed,
        total_frames,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tasks = list(TASK_SPECS) if args.task == "all" else [args.task]
    if not args.skip_download:
        for task_name in tasks:
            download_task_split(
                repo_id=args.repo_id,
                task_name=task_name,
                camera_key=args.camera_key,
                download_dir=args.download_dir,
                force_download=args.force_download,
            )

    for task_name in tasks:
        convert_single_task(task_name, args)


if __name__ == "__main__":
    main()
