import torch
import torch.nn.functional as F


class PostHocOOD:
    def __init__(self, classifier, abmil, device):
        self.classifier = classifier
        self.abmil      = abmil
        self.device     = device

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _forward(self, chief_data):
        """Shared backbone forward: returns (bag_embedding, logits, probs)."""
        bag_embedding, _ = self.abmil(chief_data)
        logits            = self.classifier(bag_embedding)
        probs             = F.softmax(logits, dim=1)
        return bag_embedding, logits, probs

    # ── OOD Methods ──────────────────────────────────────────────────────────────

    def msp(self, chief_data):
        """Maximum Softmax Probability (baseline). Higher = more ID."""
        with torch.no_grad():
            _, _, probs = self._forward(chief_data)
            scores = torch.max(probs, dim=1)[0]
        return scores

    def energy(self, chief_data, T=1.0):
        """Energy-based OOD. Higher = more ID."""
        with torch.no_grad():
            _, logits, _ = self._forward(chief_data)
            scores = T * torch.logsumexp(logits / T, dim=1)
        return scores

    def react(self, chief_data, threshold=None, energy_T=1.0):
        """
        ReAct: clip bag embedding at threshold, then compute energy score.
        Requires threshold from get_react_threshold.py.
        Higher score = more ID.
        """
        if threshold is None:
            raise ValueError("ReAct requires a threshold. Run get_react_threshold.py first.")

        with torch.no_grad():
            bag_embedding, _ = self.abmil(chief_data)
            bag_embedding     = bag_embedding.clamp(max=threshold)   # rectification
            logits            = self.classifier(bag_embedding)
            scores            = energy_T * torch.logsumexp(logits / energy_T, dim=1)
        return scores

    def mahalanobis_pp(self, chief_data, maha_params):
        """
        Mahalanobis++ OOD score.
        Features are L2-normalised before distance computation.
        Returns negative min-distance so that higher = more ID.
        """
        with torch.no_grad():
            bag_embedding, _ = self.abmil(chief_data)
            z = F.normalize(bag_embedding, p=2, dim=1)       # [B, D]

            class_means = maha_params["class_means"].to(self.device)      # [C, D]
            precision   = maha_params["precision_matrix"].to(self.device) # [D, D]

            distances = []
            for c in range(class_means.size(0)):
                diff = z - class_means[c].unsqueeze(0)
                dist = (torch.matmul(diff, precision) * diff).sum(dim=1)
                distances.append(dist)

            distances = torch.stack(distances, dim=1)        # [B, C]
            min_dist, _ = torch.min(distances, dim=1)
            scores = -min_dist                               # higher = more ID
        return scores

    def vim(self, chief_data, vim_processor):
        """
        VIM (Virtual-logit Matching) OOD score.
 
        vim_processor: VIMPostprocessor instance，必須已完成 setup()。
        Higher score → more ID.
 
        VIM 結合兩個信號：
          - Virtual logit norm：feature 落在 null-space 的幅度（越大越 OOD）
          - Free-energy score ：LogSumExp of logits（越大越 ID）
        最終 score = -vlogit * alpha + energy
        """
        return vim_processor.score(
            self.classifier, self.abmil, chief_data, device=self.device
        )

    def knn(self, chief_data, knn_processor):
        """
        KNN OOD score.
 
        knn_processor: KNNPostprocessor instance，必須已完成 fit()+load()。
        Higher score → more ID.
 
        對 test feature 做 KNN 搜尋，取第 K 個鄰居的 L2 距離取負號。
        距離越近（score 越高）→ 越接近 ID 分布。
        """
        return knn_processor.score(
            self.classifier, self.abmil, chief_data, device=self.device
        )

    def residual(self, chief_data, residual_processor):
        """
        Residual OOD score.
 
        residual_processor: ResidualPostprocessor instance，必須已完成 fit()+load()。
        Higher score → more ID.
 
        計算 feature 在 null-space 的投影 norm，取負號。
        與 VIM 的差異：無 energy 項、無 alpha 縮放，純幾何距離。
        fit 使用 val set（非 train set）。
        """
        return residual_processor.score(
            self.classifier, self.abmil, chief_data, device=self.device
        )