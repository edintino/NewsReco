import sys,os
sys.path.append(os.getcwd())

import argparse
import subprocess

def run_preprocessing():
    subprocess.run(['python3', './src/preprocessing.py'])

def run_user_embedding():
    subprocess.run(['python3', './src/user_embedding.py'])

def run_train_model():
    subprocess.run(['python3', './src/train_model.py'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different parts of the recommendation system pipeline.")
    parser.add_argument("step", choices=["preprocessing", "user_embedding", "train_model"], help="Select which step to run.")
    args = parser.parse_args()

    if args.step == "preprocessing":
        run_preprocessing()
    elif args.step == "user_embedding":
        run_user_embedding()
    elif args.step == "train_model":
        run_train_model()
