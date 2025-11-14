import torch
import pandas as pd


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
    

device = "mps" if torch.mps.is_available() else "cpu"

ratings = (
    pd.read_csv("data/animelist-dataset/users-score-2023.csv",
                usecols=['user_id','anime_id','rating'])
            .dropna()
            .query("rating > 0")
)

top_users = ratings['user_id'].value_counts().head(500).index

filtered = ratings[ratings["user_id"].isin(top_users)]
user_ids, users = pd.factorize(filtered["user_id"])
item_ids, items = pd.factorize(filtered["anime_id"])

filtered = filtered.assign(user_idx=user_ids, item_idx=item_ids)


num_users = len(users)
num_items = len(items)

model = MatrixFactorization(num_users=num_users, num_items=num_items, factors=64).to(device)
model_dict = torch.load("mf_weight.pt", map_location="cpu", weights_only=True)
model.load_state_dict(model_dict, strict=False)

# idx는 long, 모델과 같은 디바이스(mps)여야 함
idx = torch.tensor([1, 44, 12], device=device, dtype=torch.long)

# (3, 32) 임베딩
item_vecs = model.item_factors(idx)

# 1) 단순 평균 (32,)
user_vec = item_vecs.mean(dim=0)

scores = (model.item_factors.weight @ user_vec)# 예: scores[:5] -> array([7.2 , 6.8 , 6.4 , ...])

top_items = scores.detach().cpu().numpy().argsort()[::-1][:20]  # 예: top_items -> array([ 105,  320,  250, ...])
recommended_anime_ids = [items[int(idx)] for idx in top_items]  # 예: recommended_anime_ids -> [5114, 9253, 32281, ...]
print(recommended_anime_ids)  # 예시 출력: [5114, 9253, 11061, 30276, 28977, 21, 11061, 199, 6547, 22535]