import matplotlib.pyplot as plt
import numpy as np

# 假设这些是你的数据
gt_lines_fixed_num_pts = [
    np.array([[1, 1], [2, 4], [3, 9], [4, 16]]),  # 示例 3D 边界框点
    np.array([[2, 2], [3, 6], [4, 12], [5, 20]])
]
import json
gt_lines_fixed_num_pts2 = []
with open('/mnt/data/project/MapTR/test.json') as f:
    info = json.load(f)
for line in info:
    lineout = []
    for i, meta in enumerate(line['xs']):
        lineout.append([line['xs'][i], line['ys'][i]])
    gt_lines_fixed_num_pts2.append(np.array(lineout))
print(len(gt_lines_fixed_num_pts2))
gt_labels_3d = [[0, 1, 2, 0]]  # 每个点的类别标签

# 定义颜色映射（不同类别对应不同颜色）
colors_plt = ['b', 'g', 'r']  # 蓝色，绿色，红色

# 创建一个 1x4 网格的子图
fig, axs = plt.subplots(1, 5, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 4, 4, 4, 4]})

# 坐标轴设置
ax = axs[0]
start, end = -20, 40  # y 轴范围
ax.plot([0, 0], [start, end], 'k-', lw=2)  # 绘制主刻度线

# 标记刻度
for i in range(start, end + 1, 10):
    ax.plot([-0.2, 0.2], [i, i], 'k-')  # 绘制刻度线
    ax.text(-1, i, f'{i}m', ha='right', va='center')  # 绘制刻度标签

ax.set_xlim(-1, 1)
ax.axis('off')

# 输入图像1
ax = axs[1]
for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
    pts = gt_bbox_3d
    x = np.array([pt[0] for pt in pts])
    y = np.array([pt[1] for pt in pts])

    ax.plot(-y, x, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
    ax.scatter(-y, x, color=colors_plt[gt_label_3d], s=10, alpha=0.8, zorder=-1)

ax.set_title('Input Image 1')
ax.axis('off')

# 输入图像2
ax = axs[2]
for gt_bbox_3d in gt_lines_fixed_num_pts2:
    pts = gt_bbox_3d
    x = np.array([pt[0] for pt in pts])
    y = np.array([pt[1] for pt in pts])
    for pt in pts:
        if pt[0] < 0:
            print("1111111111")
    ax.plot(-y, x, color='b', linewidth=1, alpha=0.8, zorder=-1)
    # ax.scatter(-y, x, color='b', s=10, alpha=0.8, zorder=-1)

ax.set_title('Input Image 2')
ax.axis('off')

# 真值
ax = axs[3]
for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
    pts = gt_bbox_3d
    x = np.array([pt[0] for pt in pts])
    y = np.array([pt[1] for pt in pts])

    ax.plot(-y, x, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
    ax.scatter(-y, x, color=colors_plt[gt_label_3d], s=10, alpha=0.8, zorder=-1)

ax.set_title('Ground Truth')
ax.axis('off')

# 推理结果
ax = axs[4]
for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
    pts = gt_bbox_3d

    x = np.array([pt[0] for pt in pts])
    y = np.array([pt[1] for pt in pts])

    ax.plot(-y, x, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
    ax.scatter(-y, x, color=colors_plt[gt_label_3d], s=10, alpha=0.8, zorder=-1)

ax.set_title('Inference Result')
ax.axis('off')
for ax in axs:
    ax.axis('equal')
    ax.set_xlim(-25, 25)  # 根据实际数据范围调整
    ax.set_ylim(-20, 50)  # 
# 在顶部添加 meta 信息
meta_info = f"Generated on"
fig.text(0.5, 0.95, meta_info, ha='center', va='center', fontsize=12)

# 调整布局以避免重叠
fig.tight_layout()

# 显示图像
plt.savefig("test.jpg")
