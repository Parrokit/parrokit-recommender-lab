import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class RatingsDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
    
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, factors=32):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users,factors) # (80000, 32)
        self.item_factors = torch.nn.Embedding(num_items,factors) # (16471, 32)
        torch.nn.init.normal_(self.user_factors.weight, std=0.05)
        torch.nn.init.normal_(self.item_factors.weight, std=0.05)
    
    def forward(self,user_idx, item_idx):
        u = self.user_factors(user_idx) # (batch_size, 32)
        v = self.item_factors(item_idx) # (batch_size, 32)
        return (u*v).sum(dim=1) # 원소별 곱한 후 sigma{32개} -> (batch_size)
    
ratings = (
    pd.read_csv("data/animelist-dataset/users-score-2023.csv",
                usecols=['user_id','anime_id','rating'])
            .dropna()
            .query("rating > 0")
)

top_users = ratings['user_id'].value_counts().head(500).index
filtered = ratings[ratings["user_id"].isin(top_users)]

user_ids, users = pd.factorize(filtered['user_id'])
item_ids, items = pd.factorize(filtered['anime_id'])

filtered = filtered.assign(user_idx=user_ids, item_idx=item_ids)
n_users, n_items = len(users), len(items)



train_x, tmp_df = train_test_split(
    filtered[['user_idx','item_idx','rating']],
    test_size=0.3,
    random_state=42,
    stratify=filtered['user_idx']
)
valid_x, test_x = train_test_split(
    tmp_df,
    test_size=0.5,
    random_state=42,
    stratify=tmp_df['user_idx']
)


train_loader = DataLoader(RatingsDataset(train_x), batch_size=4096, shuffle=True)
valid_loader = DataLoader(RatingsDataset(valid_x), batch_size=8192)
test_loader = DataLoader(RatingsDataset(test_x), batch_size=8192)


device = "mps" if torch.mps.is_available() else "cpu"
model = MatrixFactorization(n_users, n_items, factors=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss()
train_rmse_list = []
valid_rmse_list = []

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for users_batch, items_batch, ratings_batch in train_loader:
        users_batch = users_batch.to(device)
        items_batch = items_batch.to(device)
        ratings_batch = ratings_batch.to(device).float()

        preds = model(users_batch, items_batch)
        loss = criterion(preds,ratings_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings_batch) # 배치가 다르게 들어가면 평균도 달라지는 거 막는 용

    train_rmse = (total_loss / len(train_x)) ** 0.5
    train_rmse_list.append(train_rmse)

    model.eval()
    with torch.no_grad():
        total_valid = 0.0
        for users_batch, items_batch, ratings_batch in valid_loader:
            users_batch = users_batch.to(device)
            items_batch = items_batch.to(device)
            ratings_batch = ratings_batch.to(device).float()

            preds = model(users_batch, items_batch)
            loss = criterion(preds, ratings_batch)

            total_valid += loss.item() * len(ratings_batch)
        
        valid_rmse = (total_valid / len(valid_x)) ** 0.5
        valid_rmse_list.append(valid_rmse)
    
    print(f"[Epoch: {epoch+1:03d}] train RMSE {train_rmse:.3f} | valid RMSE {valid_rmse:.3f}")


torch.save(model.state_dict(), "mf_weight.pt")