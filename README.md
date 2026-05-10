# KmeansLabel-FarthestLabel
## 声明
本仓库目前仅作为学位论文实验结果复现使用，保留一切学术权利
## 实验复现说明
### 1.安装及配置
本实验基于框架：https://github.com/aiim-research/ERASURE<br>
在此处仅将有修改或新增的部分上传<br>
配置时请先将clone基础的ERASURE项目，再将./erasure文件夹覆盖<br>
本实验的python环境已从conda中导出至./requirements.txt，精确到具体版本以确保能够一致，但实际过程中对版本应该没有太多限制<br>
其中由于使用的gpu版本的pytorch所以无法直接一键安装环境，同时注意需要和cuda版本对应

### 2.运行
./configs文件夹中的配置文件以论文章节编号，配置文件与论文章节对应<br>
如./3.5 Image data/3.5.5.cifar10_p1.jsonc对应论文3.5.5节CIFAR10数据集实验<br>
由于实验条件简陋，显存资源不足，部分实验进行了分片（如cifar10_p1、cifar10_p2、cifar10_p3），可以将配置文件中"unlearners"的内容合并，该框架可以确保其结果不受影响<br>
