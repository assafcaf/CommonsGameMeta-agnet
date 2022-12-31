import os
import sys
import gym
import time
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque
from torch.utils.data import Dataset, DataLoader

# local imports
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Utils.policies import MetaAgentAnnPolicy, MetaAgentCnnPolicy, CustomCnnNetwork
from Utils.dataset import TwoAgentsDataset
from Utils.evaluation import compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/acaftory/CommonsGame/DanfoaTest/",
        help="Root directory of the project",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Root directory of the project",
    )
    parser.add_argument(
        "--conv",
        type=bool,
        default=True,
        help="Root directory of the project",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Root directory of the project",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Relative path to dataset directory within the project ",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for supervised learning",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=float(0.8),
        help="Percentage of training vs test sizes",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Percentage of training vs test sizes",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    time_n = int(time.time())

    # load dataset from file
    dataset = TwoAgentsDataset(args.root_dir, args.dataset_dir)
    train_size = int(len(dataset) * args.train_size)

    # split dataset into train and test
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=True)

    # policy params
    policy_kwargs = dict(
        features_extractor_class=CustomCnnNetwork,
        features_extractor_kwargs=dict(features_dim=128),
    )
    item = dataset[0]
    observation_space = gym.spaces.Box(low=0, high=255, shape=item[0].shape, dtype=np.uint8)  # gym.spaces.Space
    action_space = gym.spaces.MultiDiscrete([5] * len(item[1]))  # gym.spaces.Space
    lr = lambda x: args.lr  # Callable[[float], float]

    # build policy
    if args.conv:
        policy = MetaAgentCnnPolicy(observation_space=observation_space, action_space=action_space, lr_schedule=lr,
                                    policy_kwargs=policy_kwargs)
    else:
        policy = MetaAgentAnnPolicy(observation_space=observation_space, action_space=action_space, lr_schedule=lr,
                                    policy_kwargs=policy_kwargs)
    policy.to(device)

    # set criterion
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0., 0, 0.2, 0., 0.8], dtype=torch.float32).to(device))
    accuracy = 0
    loss_ = deque(maxlen=args.print_every)
    start = time.time()

    # train policy network
    print("start training...")
    for epoch in range(args.epochs):
        for obs, actions in tqdm(train_loader, desc=f"Epoch {epoch}", ncols=75):
            # send data to device
            obs = obs.to(device)
            actions = actions.to(device=device)

            # zero the parameter gradients
            policy.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = policy.predict_logits(obs)
            loss = criterion(outputs[0], actions[:, 0]) + criterion(outputs[1], actions[:, 1])
            loss_.append(loss.item())
            loss.backward()
            policy.optimizer.step()

        accuracy = compute_accuracy(policy, test_loader, device)
        print(f"Epoch {epoch + 1}:, Loss: {np.mean(loss_):.4f},"
              f" eval acc: {accuracy:.4f}")
    print("training finished...")

    # save policy network
    print("saving model to files...")
    path = os.path.join(args.root_dir, "results/meta_supervised_models", f"model_{time_n}")
    os.mkdir(path)
    results = {"accuracy": round(accuracy.astype(float), 3),
               "loss": round(np.mean(loss_).astype(float), 3),
               "time": round(time.time() - start, 3),
               "conv": args.conv
               }

    with open(os.path.join(path, "args_parameters.json"), "w") as outfile:
        json.dump(vars(args) | results, outfile, indent=4)
    torch.save(policy.state_dict(), os.path.join(path, "meta_agent_policy.pt"))
