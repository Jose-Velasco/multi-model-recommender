import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, callbacks
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math

# Configuration
INPUT_FILE = 'train-1.txt'
OUTPUT_FILE = 'output.txt'
EMBEDDING_DIM = 32
EPOCHS = 5
BATCH_SIZE = 256
NUM_NEGATIVES = 4
TOP_K = 20         

def load_data(filename):
    print(f"Loading data from {filename}...")
    user_ids = []
    item_ids = []
    user_interaction_set = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Parsing Data"):
        parts = list(map(int, line.strip().split()))
        if not parts:
            continue
        u = parts[0]
        interacted_items = parts[1:]
        
        user_interaction_set[u] = set(interacted_items)
        
        for i in interacted_items:
            user_ids.append(u)
            item_ids.append(i)
                
    return np.array(user_ids), np.array(item_ids), user_interaction_set

def get_negatives(u_ids, i_ids, user_interaction_set, num_items, num_neg):
    print(f"Generating {num_neg} negative samples per positive...")
    negative_users = []
    negative_items = []
    
    for i in tqdm(range(len(u_ids)), desc="Negative Sampling"):
        u = u_ids[i]
        for _ in range(num_neg):
            while True:
                j = np.random.randint(num_items)
                if j not in user_interaction_set[u]:
                    negative_users.append(u)
                    negative_items.append(j)
                    break
                    
    return np.array(negative_users), np.array(negative_items)

class NDCGCallback(callbacks.Callback):

    def __init__(self, user_interacted, num_items, k=20, num_validation_users=100):
        super().__init__()
        self.user_interacted = user_interacted
        self.num_items = num_items
        self.k = k 
        self.num_val = num_validation_users
        self.ndcg_scores = []
        
        all_users = list(user_interacted.keys())
        self.val_users = np.random.choice(all_users, self.num_val, replace=False)

    def on_epoch_end(self, epoch, logs=None):
        ndcgs = []
        
        for u in self.val_users:
            if not self.user_interacted[u]:
                continue
            
            pos_item = np.random.choice(list(self.user_interacted[u]))
            
            items_to_rank = [pos_item]
            while len(items_to_rank) < 100:
                neg = np.random.randint(self.num_items)
                if neg not in self.user_interacted[u]:
                    items_to_rank.append(neg)
            
            items_to_rank = np.array(items_to_rank)
            user_input = np.full(len(items_to_rank), u)
            
            predictions = self.model.predict([user_input, items_to_rank], batch_size=100, verbose=0).flatten()
            
            ranked_indices = predictions.argsort()[::-1]
            
            rank_of_positive = np.where(ranked_indices == 0)[0][0]
            
            if rank_of_positive < self.k:
                ndcgs.append(math.log(2) / math.log(rank_of_positive + 2))
            else:
                ndcgs.append(0.0)
                
        mean_ndcg = np.mean(ndcgs)
        self.ndcg_scores.append(mean_ndcg)
        logs['val_ndcg'] = mean_ndcg

def build_ncf_model(num_users, num_items, latent_dim):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    user_embedding = layers.Embedding(num_users, latent_dim, name='user_embedding')(user_input)
    item_embedding = layers.Embedding(num_items, latent_dim, name='item_embedding')(item_input)
    
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)
    
    concat = layers.Concatenate()([user_vec, item_vec])
    
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_metrics(history, ndcg_callback, k):
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ndcg_callback.ndcg_scores, 'g-o', label=f'Validation NDCG@{k}')
    plt.title(f'NDCG@{k} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    u_train, i_train, user_interacted = load_data(INPUT_FILE)
    
    num_users = u_train.max() + 1
    num_items = i_train.max() + 1
    print(f"Stats: {num_users} Users, {num_items} Items, {len(u_train)} Interactions")

    u_neg, i_neg = get_negatives(u_train, i_train, user_interacted, num_items, NUM_NEGATIVES)
    
    train_users = np.concatenate([u_train, u_neg])
    train_items = np.concatenate([i_train, i_neg])
    train_labels = np.concatenate([np.ones(len(u_train)), np.zeros(len(u_neg))])
    
    indices = np.arange(len(train_users))
    np.random.shuffle(indices)
    train_users = train_users[indices]
    train_items = train_items[indices]
    train_labels = train_labels[indices]
    
    model = build_ncf_model(num_users, num_items, EMBEDDING_DIM)
    
    ndcg_cb = NDCGCallback(user_interacted, num_items, k=20)
    
    print("Training NCF model...")
    history = model.fit(
        [train_users, train_items], train_labels, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1,
        callbacks=[ndcg_cb],
        verbose=1
    )
    
    print("Plotting metrics...")
    plot_metrics(history, ndcg_cb, k=20)
    
    print(f"Generating top {TOP_K} recommendations...")
    all_items = np.arange(num_items)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        unique_users = sorted(list(user_interacted.keys()))
        
        for u_id in tqdm(unique_users, desc="Generating Recs"):
            user_input_arr = np.full(num_items, u_id)
            predictions = model.predict([user_input_arr, all_items], batch_size=4096, verbose=0).flatten()
            
            seen_items = list(user_interacted[u_id])
            predictions[seen_items] = -1.0 
            
            top_indices = np.argpartition(predictions, -TOP_K)[-TOP_K:]
            top_indices = top_indices[np.argsort(predictions[top_indices])[::-1]]
            
            out_line = str(u_id) + " " + " ".join(map(str, top_indices)) + "\n"
            f_out.write(out_line)

    print(f"\nDone! Recommendations saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()