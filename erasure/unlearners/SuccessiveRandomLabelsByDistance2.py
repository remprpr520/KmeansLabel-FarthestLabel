from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.core.factory_base import get_instance_kvargs
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SuccessiveRandomLabelsByDistance2(TorchUnlearner):
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

        self.info(f'Starting SRLByDistance2 with {self.epochs} epochs')

        # 计算保留数据集中每个类别的特征中心
        class_centers = {}
        class_feature_sums = [np.zeros(0) for _ in range(self.n_classes)]
        class_counts = [0 for _ in range(self.n_classes)]

        self.predictor.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.retain_set:  # 遍历 retain_set 的所有批次
                X_batch = X_batch.to(self.device)

                # 获取中间特征
                intermediate_features, _ = self.predictor.model(X_batch)

                # 将特征从GPU移到CPU，并转换为NumPy数组
                features_np = intermediate_features.cpu().numpy()
                labels_np = y_batch.cpu().numpy()

                # 累积每个类别的特征和计数
                for i, label in enumerate(labels_np):
                    label = int(label)
                    feature = features_np[i]

                    if class_feature_sums[label].size == 0:
                        class_feature_sums[label] = feature.copy()
                    else:
                        class_feature_sums[label] += feature
                    class_counts[label] += 1

        # 计算每个类别的平均特征中心
        for class_id in range(self.n_classes):
            if class_counts[class_id] > 0:
                class_centers[class_id] = class_feature_sums[class_id] / class_counts[class_id]
            else:
                self.info(f"Warning:class {class_id} not have samples.")
                # 如果某个类别没有样本，则随机初始化一个特征中心
                if len(class_feature_sums) > 0 and class_feature_sums[0].size > 0:
                    class_centers[class_id] = np.random.randn(class_feature_sums[0].size)

        # 存储 forget_set 的所有中间特征和原始标签
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

        # 对于每个遗忘数据，计算它与每个类别特征中心的距离，取最远的错误标签
        new_label_assignment = np.full(len(all_labels_np), -1, dtype=int)  # 初始化为-1

        for i, (feature, original_label) in enumerate(zip(all_features_np, all_labels_np)):
            original_label = int(original_label)

            # 计算当前样本到每个类别中心的距离
            distances = {}
            for class_id in range(self.n_classes):
                if class_id != original_label:  # 排除原始标签
                    center = class_centers[class_id]
                    distance = np.linalg.norm(feature - center)  # 使用欧几里得距离
                    distances[class_id] = distance

            if distances:  # 如果存在其他类别
                # 找到距离最远的类别作为新标签
                farthest_class = max(distances, key=distances.get)
                new_label_assignment[i] = farthest_class
            else:
                raise ValueError("not have new label.")

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
                current_batch_new_labels = new_label_assignment[start_idx:end_idx]

                # 获取当前批次的数据
                X, y = next(forget_set_iter)  # 从原始数据加载器获取批次
                X, y = X.to(self.device), y.to(self.device)

                # 将 numpy 转换为 tensor
                y_assigned = torch.tensor(current_batch_new_labels, dtype=y.dtype, device=y.device)

                # 使用预分配的 y_assigned 作为新的目标标签
                labels = y_assigned

                self.predictor.optimizer.zero_grad()
                _, output = self.predictor.model(X.to(self.device))

                # 使用新的标签计算损失
                loss = self.predictor.loss_fn(output, labels.to(self.device))
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.predictor.optimizer.step()

            # 重置 forget_set_iter
            forget_set_iter = iter(self.forget_set)

            # 用保留数据集重新训练以保持性能
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
            self.info(f'SRLByDistance2  - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')
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