import torch


class MixupCleanSelector(object):
    """
    通过“同类样本聚类 + 置信度挑选”的方式筛出更干净的样本。
    设计目标：
      - 每个样本仅与同类样本比对，避免跨类干扰。
      - 对同类样本做聚类，挑选“最有可能正确”的一个簇。
      - 若该簇的预测类别与样本标签一致，则该簇内样本视为更干净。
    """

    def __init__(
        self,
        num_clusters=2,
        max_iter=20,
        min_samples=8,
        confidence_threshold=0.6,
        device=None,
    ):
        # 聚类个数，默认 2 类：一个更干净，一个更嘈杂
        self.num_clusters = num_clusters
        # k-means 迭代次数
        self.max_iter = max_iter
        # 同类样本太少时不做聚类
        self.min_samples = min_samples
        # 选出的“最可能簇”置信度阈值
        self.confidence_threshold = confidence_threshold
        self.device = device

    def _kmeans(self, x, k):
        """
        k-means 实现
        x: [N, D]
        返回：
          - cluster_ids: [N] 每个样本所属簇
          - centroids: [k, D] 簇中心
        """
        n_samples, n_features = x.shape
        if n_samples <= k:
            # 样本太少，直接按顺序分配，保证返回可用
            cluster_ids = torch.arange(n_samples, device=x.device) % k
            centroids = torch.zeros(k, n_features, device=x.device, dtype=x.dtype)
            for i in range(k):
                mask = cluster_ids == i
                if mask.any():
                    centroids[i] = x[mask].mean(dim=0)
            return cluster_ids, centroids

        # 随机初始化簇中心
        perm = torch.randperm(n_samples, device=x.device)
        centroids = x[perm[:k]].clone()

        for _ in range(self.max_iter):
            # 计算到各中心的平方距离，取最近者
            distances = torch.cdist(x, centroids, p=2)
            cluster_ids = distances.argmin(dim=1)

            # 更新中心
            new_centroids = centroids.clone()
            for i in range(k):
                mask = cluster_ids == i
                if mask.any():
                    new_centroids[i] = x[mask].mean(dim=0)
                else:
                    # 空簇：随机重置中心
                    new_centroids[i] = x[torch.randint(0, n_samples, (1,), device=x.device)]
            # 若中心收敛则提前停止
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return cluster_ids, centroids

    def select_clean(self, features, labels, pred_probs):
        """
        输入：
          - features: [N, D] 特征向量（可来自中间层或最后一层）
          - labels: [N] 样本标签（可能含噪声）
          - pred_probs: [N, C] 模型预测概率（softmax 后）
        输出：
          - clean_mask: [N] bool，True 表示该样本被认为更干净
          - cluster_ids: [N] 对应聚类结果（仅同类内部有效）

        逻辑：
          1) 按标签分组；
          2) 对每个标签分组做 k-means；
          3) 计算每个簇的“平均预测分布”，找到最可能簇；
          4) 若该簇预测的类别与该标签一致，则该簇内样本标记为干净；
             否则本类样本都不标记为干净。
        """
        if pred_probs is None:
            raise ValueError("pred_probs 不能为空，需要模型的 softmax 概率用于判别簇的类别。")

        if self.device is None:
            device = features.device
        else:
            device = self.device

        features = features.to(device)
        labels = labels.to(device)
        pred_probs = pred_probs.to(device)

        n_samples = features.shape[0]
        clean_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
        cluster_ids = torch.full((n_samples,), -1, dtype=torch.long, device=device)

        # 遍历每个类标签
        unique_labels = torch.unique(labels)
        for lbl in unique_labels:
            idx = torch.nonzero(labels == lbl, as_tuple=False).squeeze(1)
            if idx.numel() < self.min_samples:
                # 同类样本太少，跳过聚类，保持保守策略
                continue

            feats_c = features[idx]
            probs_c = pred_probs[idx]

            # 同类样本聚类
            c_ids, _ = self._kmeans(feats_c, self.num_clusters)
            cluster_ids[idx] = c_ids

            # 计算每个簇的“平均预测分布”
            best_cluster = None
            best_score = -1.0
            best_label = None
            for k in range(self.num_clusters):
                mask = c_ids == k
                if not mask.any():
                    continue
                mean_prob = probs_c[mask].mean(dim=0)
                pred_label = mean_prob.argmax().item()
                score = mean_prob[pred_label].item()
                if score > best_score:
                    best_score = score
                    best_cluster = k
                    best_label = pred_label

            # 只有当“最可能簇”的预测类别与当前标签一致时，才认定该簇干净
            if best_cluster is not None and best_label == int(lbl) and best_score >= self.confidence_threshold:
                clean_mask[idx[c_ids == best_cluster]] = True

        return clean_mask, cluster_ids


if __name__ == "__main__":
    # 自测
    torch.manual_seed(0)
    n = 32
    d = 8
    c = 3
    feats = torch.randn(n, d)
    labels = torch.randint(0, c, (n,))
    probs = torch.softmax(torch.randn(n, c), dim=1)

    selector = MixupCleanSelector()
    clean_mask, cluster_ids = selector.select_clean(feats, labels, probs)
    print("clean ratio:", clean_mask.float().mean().item())
