import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import scipy.sparse as sp
import math
import sys
import itertools
import time
import matplotlib.pyplot as plt  

CONSTANTS = {
    'INPUT_FILE': 'train-1.txt',
    'OUTPUT_FILE': 'submission.txt',
    'PLOT_FILE': 'best_model_training.png', 
    'BATCH_SIZE': 8192,
    'EPOCHS': 50,
    'TOP_K': 20,
    'SEED': 42
}

SEARCH_SPACE = {
    'learning_rate': [0.015, 0.01, 0.005], 
    'embedding_dim': [32, 64],
    'n_layers': [2, 3]
}

def load_and_split_data(filename, validation_ratio=0.2):
    print(f"[Data] Loading {filename}...")
    users, items = [], []
    full_interactions = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if not parts: continue
            u = parts[0]
            interacted = parts[1:]
            full_interactions[u] = set(interacted)
            for i in interacted:
                users.append(u)
                items.append(i)
                
    n_users = max(users) + 1
    n_items = max(items) + 1
    
    u_train, i_train = [], []
    val_interactions = {}
    np.random.seed(CONSTANTS['SEED'])
    
    for u, user_items in full_interactions.items():
        items_list = list(user_items)
        if len(items_list) < 2:
            for i in items_list: u_train.append(u); i_train.append(i)
            continue
        np.random.shuffle(items_list)
        split_idx = int(len(items_list) * (1 - validation_ratio))
        for i in items_list[:split_idx]: u_train.append(u); i_train.append(i)
        val_interactions[u] = set(items_list[split_idx:])
            
    return {
        'n_users': n_users, 'n_items': n_items,
        'u_train': np.array(u_train, dtype=np.int32),
        'i_train': np.array(i_train, dtype=np.int32),
        'val_interactions': val_interactions,
        'full_interactions': full_interactions
    }

def build_normalized_adj(data):
    print("[Graph] Building Adjacency Matrix...")
    users = data['u_train']
    items = data['i_train']
    n_users = data['n_users']
    n_items = data['n_items']
    
    row_idx = np.concatenate([users, items + n_users])
    col_idx = np.concatenate([items + n_users, users])
    vals = np.ones_like(row_idx, dtype=np.float32)
    
    adj = sp.coo_matrix((vals, (row_idx, col_idx)), shape=(n_users+n_items, n_users+n_items))
    rowsum = np.array(adj.sum(1))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    indices = np.mat([norm_adj.row, norm_adj.col]).transpose()
    return tf.SparseTensor(indices, norm_adj.data.astype(np.float32), norm_adj.shape)

class LightGCN(models.Model):
    def __init__(self, n_users, n_items, adj, embed_dim, n_layers):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj
        self.n_layers = n_layers
        
        self.u_emb = self.add_weight(name="u_emb", shape=(n_users, embed_dim), initializer='glorot_uniform')
        self.i_emb = self.add_weight(name="i_emb", shape=(n_items, embed_dim), initializer='glorot_uniform')

    def call(self, inputs):
        user_indices, item_indices = inputs
        ego_embeddings = tf.concat([self.u_emb, self.i_emb], axis=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = tf.sparse.sparse_dense_matmul(self.adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        final_emb = tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1)
        final_u, final_i = tf.split(final_emb, [self.n_users, self.n_items], axis=0)
        
        u_emb_batch = tf.gather(final_u, user_indices)
        i_emb_batch = tf.gather(final_i, item_indices)
        return tf.reduce_sum(u_emb_batch * i_emb_batch, axis=1)
    
    def get_final_embeddings(self):
        ego_embeddings = tf.concat([self.u_emb, self.i_emb], axis=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = tf.sparse.sparse_dense_matmul(self.adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        final_emb = tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1)
        return tf.split(final_emb, [self.n_users, self.n_items], axis=0)

def evaluate(model, test_interactions, n_items, k=20):
    final_u, final_i = model.get_final_embeddings()
    final_u = final_u.numpy()
    final_i = final_i.numpy()
    
    ndcg_sum = 0
    count = 0
    
    users_to_test = list(test_interactions.keys())
    if len(users_to_test) > 500:
        np.random.shuffle(users_to_test)
        users_to_test = users_to_test[:500]
        
    for u in users_to_test:
        hidden = test_interactions[u]
        scores = final_u[u] @ final_i.T
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        
        idcg = sum([1.0/math.log2(i+2) for i in range(min(len(hidden), k))])
        dcg = sum([1.0/math.log2(i+2) for i, item in enumerate(top_k) if item in hidden])
        if idcg > 0: ndcg_sum += dcg/idcg
        count += 1
    return ndcg_sum / count

def run_experiment(data, adj, lr, dim, layers):
    print(f"\n--- Training: LR={lr}, Dim={dim}, Layers={layers} ---")
    
    model = LightGCN(data['n_users'], data['n_items'], adj, dim, layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    @tf.function
    def train_step(users, items, labels):
        with tf.GradientTape() as tape:
            preds = model([users, items])
            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_ndcg = 0
    history = {'loss': [], 'ndcg': []}
    
    for epoch in range(CONSTANTS['EPOCHS']):
        t0 = time.time()
        
        u_pos = data['u_train']
        i_pos = data['i_train']
        
        neg_items = np.random.randint(0, data['n_items'], size=len(u_pos))
        
        u_batch = np.concatenate([u_pos, u_pos])
        i_batch = np.concatenate([i_pos, neg_items])
        labels = np.concatenate([np.ones(len(u_pos)), np.zeros(len(u_pos))]).astype(np.float32)
        
        indices = np.arange(len(u_batch))
        np.random.shuffle(indices)
        
        total_loss = 0
        steps = len(indices) // CONSTANTS['BATCH_SIZE']
        
        for s in range(steps):
            batch_idx = indices[s*CONSTANTS['BATCH_SIZE'] : (s+1)*CONSTANTS['BATCH_SIZE']]
            loss = train_step(u_batch[batch_idx], i_batch[batch_idx], labels[batch_idx])
            total_loss += loss.numpy()
            
        avg_loss = total_loss / steps
        ndcg = evaluate(model, data['val_interactions'], data['n_items'])
        
        if ndcg > best_ndcg: best_ndcg = ndcg
        
        history['loss'].append(avg_loss)
        history['ndcg'].append(ndcg)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | NDCG: {ndcg:.4f} | Time: {time.time()-t0:.1f}s")
        
    return best_ndcg, model, history

def main():
    data = load_and_split_data(CONSTANTS['INPUT_FILE'])
    adj = build_normalized_adj(data)
    
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nStarting Grid Search over {len(combinations)} combinations...")
    
    best_config = None
    best_score = -1
    best_model = None
    best_history = None  
    
    results = []
    
    for config in combinations:
        score, model, history = run_experiment(
            data, adj, 
            config['learning_rate'], 
            config['embedding_dim'], 
            config['n_layers']
        )
        results.append((config, score))
        
        if score > best_score:
            best_score = score
            best_config = config
            best_model = model
            best_history = history
            
    print("\n================ FINAL RESULTS ================")
    for cfg, sc in results:
        print(f"Config: {cfg} -> Best NDCG: {sc:.4f}")
    print(f"\nWINNER: {best_config} with NDCG: {best_score:.4f}")
    
    print(f"\nGenerating plot for best model...")
    epochs_rng = range(1, CONSTANTS['EPOCHS'] + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs_rng, best_history['loss'], label='Train Loss', color='tab:red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('NDCG@20', color='tab:blue')
    ax2.plot(epochs_rng, best_history['ndcg'], label='Val NDCG', color='tab:blue', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')
    
    plt.title(f'Winner Training: LR={best_config["learning_rate"]}, Dim={best_config["embedding_dim"]}, Layers={best_config["n_layers"]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONSTANTS['PLOT_FILE'])
    print(f"Plot saved to {CONSTANTS['PLOT_FILE']}")
    plt.show()  
    
    print(f"Generating {CONSTANTS['OUTPUT_FILE']}...")
    final_u, final_i = best_model.get_final_embeddings()
    final_u = final_u.numpy()
    final_i = final_i.numpy()
    
    with open(CONSTANTS['OUTPUT_FILE'], 'w') as f:
        full = data['full_interactions']
        users = sorted(full.keys())
        for u in users:
            scores = final_u[u] @ final_i.T
            scores[list(full[u])] = -np.inf
            top_k = np.argpartition(scores, -CONSTANTS['TOP_K'])[-CONSTANTS['TOP_K']:]
            top_k = top_k[np.argsort(scores[top_k])[::-1]]
            f.write(f"{u} " + " ".join(map(str, top_k)) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()