import torch
import torch.nn as nn
import torch.nn.functional as F

# 样本筛选逻辑：同类高置信参考 + 多次 mixup + 像干净样本的比例判定。
# 为每个样本计算 p(x) = mixup 后仍像干净样本的比例，再按阈值切分 D_c / D_n。

def _cross_entropy_per_sample(logits, targets):
    # 返回每个样本的交叉熵
    return F.cross_entropy(logits, targets, reduction="none")


class MixupCleanSelector(object):
    """
    同类高置信参考 + 多次 mixup”估计每个样本的像干净样本的比例。

    logic:
    1.先用当前模型 f_theta 对所有样本前向，得到置信度/损失；
    2.每个类别取 TopK 作为参考池；
    3.对每个样本与同类参考样本做 M 次 mixup；
    4.统计 mixup 产物中“像干净”的比例 p(x)；
    5.p(x) > T 判为 clean，否则判为 noisy。
    """

    def __init__(
        self,
        pool_size=200,                  # 每类参考池大小（TopK）
        mixup_times=20,                 # 每个样本 mixup 次数（M）
        alpha=0.2,                    # Beta 分布参数（mixup 的 alpha）
        min_lambda=0.5,               # mixup 权重下界，保证产物更像原样本
        clean_metric="loss",          # clean-like 判据：loss 或 confidence

        #clean_metric="confidence",  # 这里应该是loss还是confidence？不太清楚

        loss_quantile=0.3,            # 按类 loss 分位数阈值（若未显式给 loss_threshold）
        loss_threshold=None,          # 固定 loss 阈值（优先级高于分位数）
        confidence_threshold=0.8,     # 置信度阈值（clean_metric=confidence 时使用）
        decision_threshold=0.7,       # 最终决策阈值 T：p(x) > T 判为 clean
        min_samples=2,                # 同类样本太少时不建参考池
        batch_size=128,               # 前向计算的批大小
        use_similarity_sampling=False,# 是否用“模型特征余弦相似度”选 TopM 参考样本
        similarity_batch_size=1024,  # 余弦相似度计算的 batch 大小
        device=None,
    ):
        
        self.pool_size = pool_size
        self.mixup_times = mixup_times
        self.alpha = alpha
        self.min_lambda = min_lambda
        self.clean_metric = clean_metric
        self.loss_quantile = loss_quantile
        self.loss_threshold = loss_threshold
        self.confidence_threshold = confidence_threshold
        self.decision_threshold = decision_threshold
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.use_similarity_sampling = use_similarity_sampling
        self.similarity_batch_size = similarity_batch_size
        self.device = device

    def _infer_device(self, model):
        # 从模型参数推断设备，若空模型则回落到 CPU
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move(self, tensor, device):
        # 统一的张量迁移封装
        return tensor.to(device)

    def _forward(self, model, batch):
        # 前向推理：不需要梯度
        with torch.no_grad():
            return model(batch)

    def _sample_indices(self, pool_size, num_samples, device):
        # 从参考池中随机采样 num_samples 个索引
        if pool_size >= num_samples:
            perm = torch.randperm(pool_size, device=device)
            return perm[:num_samples]
        return torch.randint(0, pool_size, (num_samples,), device=device)

    def _sample_lambdas(self, num_samples, ref_tensor):
        # 采样 mixup 权重，必要时截断到 [min_lambda, 1]
        dist = torch.distributions.Beta(self.alpha, self.alpha)
        lam = dist.sample((num_samples,)).to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if self.min_lambda is not None:
            lam = lam.clamp(min=self.min_lambda, max=1.0)
        return lam

    def _extract_features(self, model, batch):
        # 从模型获取特征向量，优先使用 extract_features
        with torch.no_grad():
            if hasattr(model, "extract_features"):
                feats = model.extract_features(batch)
            else:
                try:
                    output = model(batch, return_feat=True)
                except TypeError:
                    raise ValueError("模型需实现 extract_features 或支持 forward(return_feat=True)。")
                if isinstance(output, (tuple, list)) and len(output) == 2:
                    feats = output[1]
                else:
                    raise ValueError("forward(return_feat=True) 需返回 (logits, features)。")
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def _compute_feature_bank(self, model, inputs, device):
        # 批量提取特征并归一化，用于相似度计算
        n_samples = inputs.size(0)
        feat_chunks = []
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = inputs[start:end]
            batch = self._move(batch, device)
            feats = self._extract_features(model, batch)
            feat_chunks.append(feats.detach().cpu())
        feat_bank = torch.cat(feat_chunks, dim=0).float()
        feat_bank = F.normalize(feat_bank, p=2, dim=1)
        return feat_bank

    def _select_by_similarity(self, pool, features, index, num_samples, batch_size):
        # 使用余弦相似度从同类参考池中选 TopM
        # features: [N, D] 已归一化特征
        if pool.numel() == 0:
            return pool
        k = min(num_samples, pool.numel())
        x = features[index].reshape(1, -1)
        sims_chunks = []
        pool_size = pool.numel()
        step = max(1, int(batch_size))
        for start in range(0, pool_size, step):
            end = min(start + step, pool_size)
            chunk = pool[start:end]
            refs = features[chunk].reshape(chunk.numel(), -1)
            sims = torch.mm(refs, x.t()).squeeze(1)
            sims_chunks.append(sims)
        sims = torch.cat(sims_chunks, dim=0)
        topk = torch.topk(sims, k).indices
        return pool[topk]

    def _topk_similar(self, clean_feats, query_feats, topk, batch_size):
        # 对 query_feats 中每个样本，找出 clean_feats 中余弦相似度 TopK
        # clean_feats: [C, D], query_feats: [B, D]（均已归一化）
        if clean_feats.numel() == 0 or query_feats.numel() == 0:
            return None, None
        total_clean = clean_feats.size(0)
        k = min(topk, total_clean)
        topk_sims = None
        topk_idx = None
        step = max(1, int(batch_size))
        for start in range(0, total_clean, step):
            end = min(start + step, total_clean)
            chunk = clean_feats[start:end]
            sims = torch.mm(chunk, query_feats.t())  # [chunk, B]
            chunk_k = min(k, sims.size(0))
            chunk_topk_sims, chunk_topk_idx = torch.topk(sims, chunk_k, dim=0)
            chunk_topk_idx = chunk_topk_idx + start
            if topk_sims is None:
                topk_sims = chunk_topk_sims
                topk_idx = chunk_topk_idx
                continue
            combined_sims = torch.cat([topk_sims, chunk_topk_sims], dim=0)
            combined_idx = torch.cat([topk_idx, chunk_topk_idx], dim=0)
            keep_k = min(k, combined_sims.size(0))
            topk_sims, topk_pos = torch.topk(combined_sims, keep_k, dim=0)
            topk_idx = torch.gather(combined_idx, 0, topk_pos)
        return topk_sims, topk_idx

    def _compute_base_stats(self, model, inputs, labels, device):
        # 全量前向，计算每个样本的置信度和loss
        n_samples = inputs.size(0) 
        conf = torch.zeros(n_samples)
        loss = torch.zeros(n_samples) if self.clean_metric == "loss" else None
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples) # 批结束索引
            batch = inputs[start:end]
            batch = self._move(batch, device)
            logits = self._forward(model, batch) 
            probs = F.softmax(logits, dim=1)
            conf[start:end] = probs.max(dim=1).values.detach().cpu()
            if self.clean_metric == "loss":
                targets = labels[start:end].long()
                targets = self._move(targets, device)
                loss_batch = _cross_entropy_per_sample(logits, targets)
                loss[start:end] = loss_batch.detach().cpu()
        return conf, loss

    def _build_pools(self, labels, conf):
        # 按类别建立 TopK 参考池
        unique_labels = torch.unique(labels)
        pool_by_class = {}
        for lbl in unique_labels: 
            idx = (labels == lbl).nonzero(as_tuple=True)[0] 
            # 这里让ai写的看不懂实现有点复杂，用常规数据类型会出错
            # 上面找出 labels 中等于标量 lbl 的所有索引
            if idx.numel() < self.min_samples: 
                continue
            conf_lbl = conf[idx] 
            k = min(self.pool_size, idx.numel()) 
            if k == 0:
                continue
            topk = torch.topk(conf_lbl, k).indices 
            pool_by_class[int(lbl.item())] = idx[topk] 
        return pool_by_class

    def _build_loss_thresholds(self, labels, loss):
        # 按类别计算 loss 阈值（分位数或固定阈值）
        unique_labels = torch.unique(labels)
        loss_thresholds = {}
        for lbl in unique_labels:
            idx = (labels == lbl).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if self.loss_threshold is not None:
                loss_thresholds[int(lbl.item())] = float(self.loss_threshold)
                continue
            loss_vals = loss[idx]
            q = torch.quantile(loss_vals, self.loss_quantile)
            loss_thresholds[int(lbl.item())] = float(q.item())
        return loss_thresholds

    def select_clean(self, inputs, labels, model):
        """
        Args:
            inputs: Tensor [N, ...] 输入样本（图像或特征）。
            labels: Tensor [N] 标签（可能含噪）。
            model: 当前模型 f_theta，用于打分与判别。
        Returns:
            clean_mask: Tensor [N] bool，True 表示判为 clean。
            clean_scores: Tensor [N] float，p(x) clean-like 比例。
        """

        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("输入和标签需要为 torch.Tensor 类型。")
        if self.clean_metric not in ("loss", "confidence"):
            raise ValueError("clean_metric 必须为 'loss' 或 'confidence'。")

        # 暂时切 eval，避免 BN/Dropout 影响，这里搞了个标志位记录下之前的状态方便restore
        model_was_training = model.training 
        model.eval() 

        device = self._infer_device(model)
        # 统一在 CPU 上索引与构建参考池，避免 GPU 内存占用
        inputs_cpu = inputs.detach().cpu().float()
        labels_cpu = labels.detach().cpu().long()

        # 1) 全量前向：得到每个样本的置信度/损失
        conf, loss = self._compute_base_stats(model, inputs_cpu, labels_cpu, device)
        # 2) 按类建立参考池（TopK）
        pool_by_class = self._build_pools(labels_cpu, conf)
        feature_bank = None
        if self.use_similarity_sampling:
            # 额外提取特征，供相似度采样使用
            feature_bank = self._compute_feature_bank(model, inputs_cpu, device)
        loss_thresholds = None
        if self.clean_metric == "loss":
            # 3) 按类计算 loss 判据阈值
            loss_thresholds = self._build_loss_thresholds(labels_cpu, loss)

        n_samples = inputs_cpu.size(0)
        clean_scores = torch.zeros(n_samples)
        for i in range(n_samples):
            y = int(labels_cpu[i])
            pool = pool_by_class.get(y)
            if pool is None or pool.numel() == 0:
                # 同类参考池为空，无法评估，保持 0
                continue
            if pool.numel() > 1:
                # 避免把自己当参考（若被包含）
                pool = pool[pool != i]
                if pool.numel() == 0:
                    pool = pool_by_class.get(y)
            # 4) 从同类参考池中选择 M 个（随机 or 余弦相似度 TopM）
            if self.use_similarity_sampling:
                ref_idx = self._select_by_similarity(
                    pool,
                    feature_bank,
                    i,
                    self.mixup_times,
                    self.similarity_batch_size,
                )
                mixup_count = ref_idx.numel()
            else:
                pick = self._sample_indices(pool.numel(), self.mixup_times, pool.device)
                ref_idx = pool[pick]
                mixup_count = self.mixup_times
            x = inputs_cpu[i]
            refs = inputs_cpu[ref_idx]

            # 5) 生成 mixup 产物
            if mixup_count == 0:
                continue
            lam = self._sample_lambdas(mixup_count, x)
            lam = lam.view(mixup_count, *([1] * x.dim()))
            x_exp = x.unsqueeze(0).expand_as(refs)
            mixup = lam * x_exp + (1.0 - lam) * refs

            # 6) 判别 mixup 是否 clean-like
            mixup = self._move(mixup, device)
            logits = self._forward(model, mixup)
            if self.clean_metric == "confidence":
                probs = F.softmax(logits, dim=1)
                conf_mix = probs.max(dim=1).values
                clean_like = conf_mix >= self.confidence_threshold
            else:
                targets = torch.full((mixup_count,), y, device=device, dtype=torch.long)
                loss_mix = _cross_entropy_per_sample(logits, targets)
                clean_like = loss_mix <= loss_thresholds.get(y, float("inf"))

            # 7) p(x) = clean-like 比例
            score = clean_like.float().mean()
            clean_scores[i] = score.detach().cpu().item()

        # 8) 按阈值切分 clean / noisy
        clean_mask = clean_scores > self.decision_threshold
        model.train(model_was_training)
        return clean_mask, clean_scores

    def correct_noisy_labels(
        self,
        inputs,
        labels,
        model,
        clean_mask,
        topk=10,
        num_classes=None,
        noisy_batch_size=None,
    ):
        """
        对噪声样本进行标签修正：
        - 对每个噪声样本，与所有干净样本做余弦相似度；
        - 取 TopK 相似的干净样本；
        - 按相似度对其标签做加权投票，得到 one-hot 伪标签。

        Returns:
            corrected_labels: Tensor [N]，噪声样本被修正后的标签（CPU）。
            noisy_onehot: Tensor [N_noisy, C]，噪声样本的 one-hot 伪标签。
            noisy_idx: Tensor [N_noisy]，噪声样本索引。
        """
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("输入和标签需要为 torch.Tensor 类型。")
        if clean_mask is None or not isinstance(clean_mask, torch.Tensor):
            raise TypeError("clean_mask 需要为 torch.Tensor 类型。")

        if noisy_batch_size is None:
            noisy_batch_size = self.batch_size

        device = self._infer_device(model)
        inputs_cpu = inputs.detach().cpu().float()
        labels_cpu = labels.detach().cpu().long()
        clean_mask_cpu = clean_mask.detach().cpu().bool()

        if num_classes is None:
            num_classes = int(labels_cpu.max().item()) + 1

        clean_idx = clean_mask_cpu.nonzero(as_tuple=True)[0]
        noisy_idx = (~clean_mask_cpu).nonzero(as_tuple=True)[0]
        corrected_labels = labels_cpu.clone()

        if clean_idx.numel() == 0 or noisy_idx.numel() == 0:
            return corrected_labels, torch.zeros((0, num_classes)), noisy_idx

        # 1) 计算特征库（已归一化）
        feature_bank = self._compute_feature_bank(model, inputs_cpu, device)
        clean_feats = feature_bank[clean_idx]
        clean_labels = labels_cpu[clean_idx]

        # 2) 对噪声样本批量做 TopK 相似度搜索并投票
        noisy_onehot = torch.zeros((noisy_idx.numel(), num_classes))
        for start in range(0, noisy_idx.numel(), noisy_batch_size):
            end = min(start + noisy_batch_size, noisy_idx.numel())
            batch_idx = noisy_idx[start:end]
            noisy_feats = feature_bank[batch_idx]

            topk_sims, topk_idx = self._topk_similar(
                clean_feats,
                noisy_feats,
                topk,
                self.similarity_batch_size,
            )
            if topk_sims is None:
                continue

            # topk_idx: [K, B] -> 对应 clean_labels 的索引
            topk_labels = clean_labels[topk_idx]

            # 按相似度加权投票，得到每个类别的得分
            scores = torch.zeros((noisy_feats.size(0), num_classes))
            scores.scatter_add_(1, topk_labels.t(), topk_sims.t())
            pred = scores.argmax(dim=1)

            corrected_labels[batch_idx] = pred
            noisy_onehot[start:end] = F.one_hot(pred, num_classes=num_classes).float()

        return corrected_labels, noisy_onehot, noisy_idx


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 32
    c = 3
    h = 4
    w = 4
    inputs = torch.randn(n, 1, h, w)
    labels = torch.randint(0, c, (n,))

    class TinyModel(nn.Module):
        def __init__(self, in_dim, num_classes):
            super(TinyModel, self).__init__()
            self.fc = nn.Linear(in_dim, num_classes)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    model = TinyModel(h * w, c)
    selector = MixupCleanSelector()
    clean_mask, clean_scores = selector.select_clean(inputs, labels, model)
    print("clean ratio:", clean_mask.float().mean())
