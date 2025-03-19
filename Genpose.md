这篇文章（"GenPose: Generative Category-level Object Pose Estimation via Diffusion Models"）提出了一种利用扩散模型（Diffusion Model）从部分观测的点云估计物体6D姿态的新方法。以下是详细说明其输入、输出以及如何使用扩散模型完成姿态估计的过程：

### 输入与输出
- **输入**：部分观测的物体点云（Partially Observed Point Cloud）。在实际场景中，由于遮挡或视角限制，传感器通常只能捕获物体的部分几何信息，因此输入是一个不完整的3D点云，表示为 \( O_i \in \mathbb{R}^{3 \times N} \)（其中 \( N \) 是点云中的点数，例如文中提到使用1024个点）。
- **输出**：物体的6D姿态（6D Pose），包括旋转（Rotation）和平移（Translation）。6D姿态描述了物体在三维空间中的位置和方向，旋转部分用连续的6维表示法表示（避免四元数或欧拉角的不连续性），平移部分用3维向量表示，最终输出为 \( \hat{p} \in SE(3) \)。

### 使用扩散模型估计姿态的过程
该方法将物体姿态估计问题重新定义为条件生成建模（Conditional Generative Modeling），通过结合得分基础扩散模型（Score-based Diffusion Model）和能量基础模型（Energy-based Model），从部分点云生成并优化多个可能的姿态假设，最终聚合得到单一的姿态估计。以下是具体步骤：

#### 1. 问题建模
- 传统方法通常通过回归直接预测单一姿态，但对于对称物体或部分观测点云，可能存在多个合理的姿态假设（多假设问题，Multi-hypothesis Issue）。因此，作者提出通过生成模型来估计条件姿态分布 \( p_{\text{data}}(\boldsymbol{p} \mid O) \)，即给定点云 \( O \) 的所有可能姿态分布。

#### 2. 得分基础扩散模型（Score-based Diffusion Model）
- **训练阶段**：
  - 使用训练数据集 \( \mathcal{D} = \{(\boldsymbol{p}_i, O_i)\}_{i=1}^n \)（包含点云和对应的真实姿态对）。
  - 通过扩散过程逐步向真实姿态 \( \boldsymbol{p}(0) \) 添加噪声，生成一系列噪声扰动后的姿态 \( \boldsymbol{p}(t) \)，其中 \( t \in [0,1] \) 表示时间步长，噪声水平由 \( \sigma(t) \) 控制。
  - 训练一个得分网络 \( \Phi_\theta(\boldsymbol{p}(t), t \mid O) \) 来估计扰动姿态分布的得分函数（即对数密度的梯度 \( \nabla_{\boldsymbol{p}} \log p_t(\boldsymbol{p} \mid O) \)），通过去噪得分匹配（Denoising Score Matching, DSM）损失函数优化：
    \[
    \mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(\epsilon, 1)} \left\{ \lambda(t) \mathbb{E} \left[ \left\| \Phi_\theta(\boldsymbol{p}(t), t \mid O) - \frac{\boldsymbol{p}(0) - \boldsymbol{p}(t)}{\sigma(t)^2} \right\|_2^2 \right] \right\}
    \]
  - 得分网络的输入包括扰动姿态 \( \boldsymbol{p}(t) \)、时间步 \( t \) 和点云 \( O \)，输出为姿态空间中的得分向量。

- **测试阶段**：
  - 给定新的点云 \( O^* \)，从噪声分布（如高斯分布）开始，通过逆扩散过程采样多个姿态候选 \( \{\hat{\boldsymbol{p}}_i\}_{i=1}^K \)。
  - 使用训练好的得分网络 \( \Phi_\theta \) 指导采样，具体通过求解概率流常微分方程（Probability Flow ODE）实现：
    \[
    \frac{d \boldsymbol{p}}{d t} = -\sigma(t) \dot{\sigma}(t) \Phi_\theta(\boldsymbol{p}, t \mid O)
    \]
  - 采样结果是从条件分布 \( p_{\text{data}}(\boldsymbol{p} \mid O^*) \) 中生成的多个可能姿态。

#### 3. 能量基础模型（Energy-based Model）优化
- **问题**：扩散模型采样的姿态候选中可能包含低概率的离群值（Outliers），直接聚合这些候选会导致性能下降。
- **解决方案**：
  - 训练一个能量网络 \( \Psi_\phi(\boldsymbol{p}, t \mid O) \) 来估计每个候选姿态的似然（Likelihood）。能量网络通过监督其梯度与得分网络输出一致来训练，避免了直接计算似然的复杂积分过程：
    \[
    \mathcal{L}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(\epsilon, 1)} \left\{ \lambda(t) \mathbb{E} \left[ \left\| \nabla_{\boldsymbol{p}(t)} \Psi_\phi(\boldsymbol{p}(t), t \mid O) - \frac{\boldsymbol{p}(0) - \boldsymbol{p}(t)}{\sigma(t)^2} \right\|_2^2 \right] \right\}
    \]
  - 能量网络的输出 \( \Psi_\phi(\hat{\boldsymbol{p}}_i, \epsilon \mid O^*) \) 近似表示候选姿态的对数似然，用于对 \( K \) 个候选进行排序。
  - 保留高似然的候选（例如前 \( \delta\% \)，文中默认 \( \delta = 60\% \)），过滤掉低似然的离群值。

#### 4. 候选聚合
- 对剩余的高似然候选 \( \{\hat{\boldsymbol{p}}_{\tau_i}\}_{i=1}^M \)（其中 \( M = \lfloor \delta \cdot K \rfloor \)）进行聚合：
  - **平移部分**：通过简单均值池化（Mean Pooling）计算平均平移 \( \hat{T} = \frac{\sum_{i=1}^M \hat{T}_{\tau_i}}{M} \)。
  - **旋转部分**：由于旋转不在欧几里得空间中，需特殊处理。将旋转转换为四元数 \( \{\hat{\boldsymbol{q}}_{\tau_i}\}_{i=1}^M \)，通过求解最大特征值问题计算平均四元数：
    \[
    \hat{\boldsymbol{q}} = \underset{\boldsymbol{q} \in SO(3)}{\arg \max} \boldsymbol{q}^T \left( \frac{\sum_{i=1}^M A(\hat{\boldsymbol{q}}_{\tau_i})}{M} \right) \boldsymbol{q}, \quad A(\hat{\boldsymbol{q}}_{\tau_i}) = \hat{\boldsymbol{q}}_{\tau_i} \hat{\boldsymbol{q}}_{\tau_i}^T
    \]
  - 最终输出估计姿态 \( \hat{\boldsymbol{p}} = (\hat{T}, \hat{R}) \)。

### 总结
该方法通过以下步骤从输入的部分点云估计物体6D姿态：
1. 使用得分基础扩散模型生成多个姿态候选，解决多假设问题。
2. 借助能量基础模型估计候选的似然，过滤掉低概率的离群值。
3. 对高似然候选进行聚合（平移均值池化，旋转四元数平均），得到最终姿态估计。

这种方法不依赖特定类别的先验知识（Prior-free），在对称物体和部分观测场景中表现出色，并在 REAL275 数据集上取得了 state-of-the-art 的性能。