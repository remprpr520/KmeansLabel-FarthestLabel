from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.core.factory_base import get_instance_kvargs
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

class SuccessiveRandomLabelsPlus2(TorchUnlearner):
    def init(self):
        """
        Initializes the SuccessiveRandomLabels class with global and local contexts.
        """
        super().init()
        self.epochs = self.local.config['parameters']['epochs']
        self.epochs_re = self.local.config['parameters']['epochs_re']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.task = self.local.config['parameters']['task']
        self.predictor.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {
                'params': self.predictor.model.parameters(),
                **self.local_config['parameters']['optimizer']['parameters']
            }
        )
        self.retain_set, _ = self.dataset.get_loader_for(self.ref_data_retain)
        self.forget_set, _ = self.dataset.get_loader_for(self.ref_data_forget)
        self.n_classes = self.dataset.n_classes

    def __unlearn__(self):
        """
        Fine-tunes the model with both the retain set and forget set. The labels for the forget set are randomly assigned and different from the original ones.
        """

        self.info(f'Starting SRLPlus with {self.epochs} epochs')

        # 收集 forget_set 的所有中间特征、原始标签和索引
        all_forget_features = []
        all_original_labels = []
        batch_info = []

        self.predictor.model.eval()
        with torch.no_grad():
            current_index = 0
            for X_batch, y_batch in self.forget_set:  # 遍历 forget_set 的所有批次
                X_batch = X_batch.to(self.device)

                # 获取中间特征
                intermediate_features, _ = self.predictor.model(X_batch)

                all_forget_features.append(intermediate_features.cpu().numpy())
                all_original_labels.append(y_batch.cpu().numpy())

                # 记录当前批次的信息
                batch_size = X_batch.size(0)
                batch_info.append({
                    'start_idx': current_index,
                    'end_idx': current_index + batch_size,
                    'original_labels': y_batch.cpu().numpy(),  # 存储原始标签
                    'batch_size': batch_size
                })
                current_index += batch_size

        # 将所有批次的特征和标签堆叠成一个大矩阵
        all_features_np = np.vstack(all_forget_features)
        all_labels_np = np.hstack(all_original_labels)

        # 按原始标签对数据进行分组，并对每组进行聚类
        cluster_assignment_map = np.full(len(all_labels_np), -1, dtype=int)  # 用于存储每个样本的新标签

        for class_id in range(self.n_classes):
            # 找到属于当前类别的样本索引
            class_mask = (all_labels_np == class_id)
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                self.info(f"Warning:class {class_id} not have samples.")
                continue

            # 获取当前类别的特征
            class_features = all_features_np[class_mask]

            # 为当前类别聚类成 n_classes - 1 个簇
            n_sub_clusters = self.n_classes - 1

            if len(class_features) < n_sub_clusters:
                # 如果样本数少于聚类数，调整聚类数
                n_sub_clusters = len(class_features)

            if n_sub_clusters > 0:
                kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
                sub_cluster_labels = kmeans.fit_predict(class_features)

                # 为目标标签创建一个随机的映射，不包含原始类别 class_id
                possible_target_labels = [c for c in range(self.n_classes) if c != class_id]

                # 为每个簇随机分配一个目标标签
                assigned_targets = {}
                for sub_cluster_id in range(n_sub_clusters):
                    # 随机选择一个不同于原始类别 class_id 的标签
                    assigned_target = np.random.choice(possible_target_labels)
                    assigned_targets[sub_cluster_id] = assigned_target

                # 保存对应的新标签
                for i, original_idx in enumerate(class_indices):
                    sub_cluster_of_sample = sub_cluster_labels[i]
                    new_target_label = assigned_targets[sub_cluster_of_sample]
                    cluster_assignment_map[original_idx] = new_target_label

        # 主训练循环
        forget_set_iter = iter(self.forget_set)   # 重新定义一个迭代器与 batch_info 对齐

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            # 遍历 batch_info 来同步获取批次数据和对应的预分配新标签
            for batch_meta in batch_info:
                start_idx = batch_meta['start_idx']
                end_idx = batch_meta['end_idx']

                # 获取当前批次对应的预分配新标签
                current_batch_new_labels = cluster_assignment_map[start_idx:end_idx]

                # 获取当前批次的数据
                X, y = next(forget_set_iter)  # 从原始数据加载器获取批次
                X, y = X.to(self.device), y.to(self.device)

                # 将 numpy 转换为 tensor
                y_preassigned = torch.tensor(current_batch_new_labels, dtype=y.dtype, device=y.device)

                # 直接使用预分配的 y_preassigned 作为新的目标标签
                labels = y_preassigned

                self.predictor.optimizer.zero_grad()
                _, output = self.predictor.model(X.to(self.device))

                # 使用新的标签计算损失
                loss = self.predictor.loss_fn(output, labels.to(self.device))
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.predictor.optimizer.step()

            forget_set_iter = iter(self.forget_set)

            for epoch_re in range(self.epochs_re):
                for X, labels in self.retain_set:
                    X, labels = X.to(self.device), labels.to(self.device)
                    self.predictor.optimizer.zero_grad()
                    _, output = self.predictor.model(X.to(self.device))
                    loss = self.predictor.loss_fn(output, labels.to(self.device))
                    losses.append(loss.to('cpu').detach().numpy())
                    loss.backward()
                    self.predictor.optimizer.step()

            # 清理缓存
            del output, loss, X, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_loss = sum(losses) / len(losses)
            self.info(f'SRLPlus - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')
            self.predictor.lr_scheduler.step()

        return self.predictor

    def check_configuration(self):
        super().check_configuration()
        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain",
                                                                                                 'retain')
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget",
                                                                                                 'forget')
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer",
                                                                                           {'class': 'torch.optim.Adam',
                                                                                            'parameters': {}})
        self.local.config['parameters']['task'] = self.local.config['parameters'].get("task", 'auto')