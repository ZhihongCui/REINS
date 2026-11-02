import argparse
import sys
import os

# Add project root to path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline.train import train_pipeline
from src.pipeline.rule_inference import InferenceRunner  # self-evolving runtime
import gymnasium as gym
from config.config import CONFIG
import json
from src.utils.visualization import visualize_inference_results
from gymnasium.wrappers import RecordVideo

def main(args):
    if args.stage == "train":
        print(">>> [Stage: TRAIN] Starting structure function and rule scorer training")
        train_pipeline(num_train_episodes=args.num_train_episodes)

    elif args.stage == "inference":
        print(">>> [Stage: INFERENCE] Starting causal graph construction and rule inference")
        ###
        config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["x", "y", "vx", "vy"],
        },
        "duration": 200
        }
        env = gym.make(CONFIG["env_name"], render_mode="rgb_array", config=config)
        env = env.unwrapped
        obs, _ = env.reset()

        env = RecordVideo(
        env,
        video_folder="result/videos",
        name_prefix="episode",
        episode_trigger=lambda ep: ep < 3,
        )
        ###
        runner = InferenceRunner(env, config_path=args.config)
        results= runner.run_multiple(n_episodes=args.num_test_episodes)
        result_dir = "result/inference"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "inference_results.json")

        print(f"[DEBUG] Preparing to write results to {result_path}, len(results)={len(results)}")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Inference results saved to {result_path}")

        visualize_inference_results(result_path, output_dir=result_dir)
        print("[OK] Visualization images generated")
        exit(0)

    elif args.stage == "all":
        print(">>> [Stage: TRAIN] Starting structure function and rule scorer training")
        train_pipeline(num_train_episodes=args.num_train_episodes)

        print(">>> [Stage: INFERENCE] Starting causal graph construction and rule inference")

        env = gym.make("highway-v0")
        env.configure({
            "duration": 200,
            "policy_frequency": 15,
            "vehicles_count": 8
        })

        runner = InferenceRunner(env, config_path=args.config)
        runner.run_multiple(n_episodes=args.num_test_episodes)

    else:
        raise ValueError("stage must be 'train', 'inference', or 'all'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/inference pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["train", "inference", "all"],
                        help="Select stage: train, inference, all")
    parser.add_argument("--num_train_episodes", type=int, default=30,
                        help="Number of sampling episodes for training")
    parser.add_argument("--num_test_episodes", type=int, default=3,
                        help="Number of episodes for testing")
    parser.add_argument("--config", type=str, default="config/inference_runner.yaml")

    args = parser.parse_args()

    main(args)
