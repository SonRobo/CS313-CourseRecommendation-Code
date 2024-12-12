import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT


def evaluate_for_user(model, user_id, dataloader, device, top_k=10):
    """
    Evaluate the model for a single user and return the top_k predictions.
    """
    model.eval()

    # Convert user_id and item_ids to tensors
    user_tensor = torch.LongTensor([user_id]).to(device)
    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    # Predict scores for all items for the given user
    with torch.no_grad():
        scores = model(user_tensor, item_ids, mode='predict')  # (1, n_items)

    scores = scores.cpu().numpy().flatten()

    # Exclude items the user has already interacted with
    train_user_dict = dataloader.train_user_dict
    interacted_items = set(train_user_dict.get(user_id, []))

    # Create a list of (item_id, score), filter out interacted items, and sort by score
    item_scores = [(item_id, score) for item_id, score in enumerate(scores) if item_id not in interacted_items]
    item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)

    # Get the top_k items
    top_items = item_scores[:top_k]
    return top_items


def predict_for_user(args, user_id):
    """
    Predict top 10 recommendations for a single user.
    """
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = DataLoaderKGAT(args, logging)

    # Load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # Predict top 10 items for the user
    top_k = 10
    top_items = evaluate_for_user(model, user_id, data, device, top_k)

    print(f"Top {top_k} recommendations for user {user_id}:")
    for rank, (item_id, score) in enumerate(top_items, start=1):
        print(f"{rank}: Item {item_id} with score {score}")

    return top_items




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, default="")
    parser.add_argument('--use_pretrain', type=int, default=0)
    parser.add_argument('--pretrain_model_path', type=str, default="")
    parser.add_argument('--n_epoch', type=int, default=6)
    parser.add_argument('--cf_batch_size', type=int, default=1024)
    parser.add_argument('--kg_batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--cf_print_every', type=int, default=500)
    parser.add_argument('--kg_print_every', type=int, default=50)
    parser.add_argument('--evaluate_every', type=int, default=2)
    parser.add_argument('--Ks', type=str, default="[1, 5, 10]")
    parser.add_argument('--save_dir', type=str, default="./results/")
    parser.add_argument('--user_id', type=int, required=True, help="User ID to predict for")  # Thêm tham số user_id
    parser.add_argument('--pretrain_embedding_dir', type=str)  
    parser.add_argument('--laplacian_type', type=str, default='random-walk')  
    args = parser.parse_args()
    
    predict_for_user(args, user_id=args.user_id)


