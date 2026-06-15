
# coding: utf-8

# In[11]:


# 导入
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


DATA_ROOT = Path(r"E:\桌面\各种作业的东西\人工智能微专业\计算机视觉基础\实验 阶段2\交大视觉印象数据集2026")


# In[3]:


retrieval_root = DATA_ROOT / "image_retrieval"
base_dir = retrieval_root / "base"
query_dir = retrieval_root / "query"

det_data_dir = DATA_ROOT / "object_detection" / "data"

output_root = Path("outputs")
pk_curve_dir = output_root / "pk_curves"
text_det_dir = output_root / "text_detection_detectonly_final"
feature_dir = output_root / "features"

for d in [output_root, pk_curve_dir, text_det_dir, feature_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("base_dir:", base_dir.exists(), base_dir)
print("query_dir:", query_dir.exists(), query_dir)
print("det_data_dir:", det_data_dir.exists(), det_data_dir)


# In[4]:


# 读取图片路径，定义标签函数
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

def get_label(path):
    return path.stem.split("-")[0]

base_imgs = sorted([p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
query_imgs = sorted([p for p in query_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
det_img_paths = sorted([p for p in det_data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

print("base 图片数量:", len(base_imgs))
print("query 图片数量:", len(query_imgs))
print("文字检测 data 图片数量:", len(det_img_paths))

print("前5个 query:", [p.name for p in query_imgs[:5]])
print("前5个 base:", [p.name for p in base_imgs[:5]])


# In[5]:


#过滤了一张坏图
def filter_valid_images(paths):
    valid = []
    bad = []

    for p in paths:
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except Exception:
            bad.append(p)

    return valid, bad

base_imgs, bad_base_imgs = filter_valid_images(base_imgs)
query_imgs, bad_query_imgs = filter_valid_images(query_imgs)

print("有效 base:", len(base_imgs))
print("坏 base:", len(bad_base_imgs))
print("有效 query:", len(query_imgs))
print("坏 query:", len(bad_query_imgs))

if bad_base_imgs:
    print("坏图示例:", [p.name for p in bad_base_imgs[:10]])


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"

print("torch version:", torch.__version__)
print("device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

print("ResNet50 feature extractor loaded.")


# In[7]:


@torch.no_grad()
def extract_features(paths, batch_size=64):
    ds = ImagePathDataset(paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats = []
    out_paths = []

    for imgs, batch_paths in dl:
        imgs = imgs.to(device)
        f = model(imgs)
        f = torch.nn.functional.normalize(f, dim=1)

        feats.append(f.cpu().numpy())
        out_paths.extend(batch_paths)

    feats = np.concatenate(feats, axis=0)
    out_paths = [Path(p) for p in out_paths]

    return feats, out_paths


base_feat_file = feature_dir / "base_feats.npy"
query_feat_file = feature_dir / "query_feats.npy"
base_path_file = feature_dir / "base_paths.npy"
query_path_file = feature_dir / "query_paths.npy"

if base_feat_file.exists() and query_feat_file.exists() and base_path_file.exists() and query_path_file.exists():
    print("发现已有特征文件，直接加载。")
    base_feats = np.load(base_feat_file)
    query_feats = np.load(query_feat_file)
    base_paths = [Path(p) for p in np.load(base_path_file, allow_pickle=True)]
    query_paths = [Path(p) for p in np.load(query_path_file, allow_pickle=True)]
else:
    print("开始提取图像特征。")
    base_feats, base_paths = extract_features(base_imgs, batch_size=64)
    query_feats, query_paths = extract_features(query_imgs, batch_size=64)

    np.save(base_feat_file, base_feats)
    np.save(query_feat_file, query_feats)
    np.save(base_path_file, np.array([str(p) for p in base_paths], dtype=object))
    np.save(query_path_file, np.array([str(p) for p in query_paths], dtype=object))

print("base_feats:", base_feats.shape)
print("query_feats:", query_feats.shape)


# In[8]:


similarity = query_feats @ base_feats.T
print("similarity shape:", similarity.shape)

idx = 0
top5 = similarity[idx].argsort()[-5:][::-1]

print("Query:", query_paths[idx].name)
print("Top-5 retrieved images:")

for rank, i in enumerate(top5, 1):
    print(rank, base_paths[i].name)


# In[12]:


query_labels = np.array([get_label(p) for p in query_paths])
base_labels = np.array([get_label(p) for p in base_paths])

Ks = [1, 5, 10, 20, 40, 60]

results = {lm: [] for lm in LANDMARKS}

for lm in LANDMARKS:
    q_indices = np.where(query_labels == lm)[0]

    for K in Ks:
        ps = []

        for qi in q_indices:
            topk = similarity[qi].argsort()[-K:][::-1]
            correct = np.sum(base_labels[topk] == query_labels[qi])
            ps.append(correct / K)

        results[lm].append(np.mean(ps) if len(ps) > 0 else np.nan)

results_df = pd.DataFrame(results, index=[f"P@{k}" for k in Ks]).T
results_df


# In[13]:


for lm in LANDMARKS:
    plt.figure(figsize=(4, 3))
    plt.plot(Ks, results[lm], marker="o")
    plt.title(f"P@K - {lm}")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid(True)

    save_path = pk_curve_dir / f"P@K_{lm}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

results_df.to_csv(pk_curve_dir / "retrieval_pk_results.csv", encoding="utf-8-sig")

print("P@K 曲线已保存到:", pk_curve_dir.resolve())
print("共保存:", len(list(pk_curve_dir.glob('P@K_*.png'))), "张图")


# In[14]:


sample_curve_paths = sorted(pk_curve_dir.glob("P@K_*.png"))[:4]

plt.figure(figsize=(12, 8))

for i, p in enumerate(sample_curve_paths, 1):
    img = Image.open(p)
    plt.subplot(2, 2, i)
    plt.imshow(img)
    plt.axis("off")
    plt.title(p.stem)

plt.tight_layout()
plt.show()


# In[15]:


import easyocr

reader = easyocr.Reader(["ch_sim", "en"], gpu=torch.cuda.is_available())

print("EasyOCR loaded.")


# In[16]:


def draw_easyocr_detect_only(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    horizontal_list, free_list = reader.detect(img_np)

    draw = ImageDraw.Draw(img)
    count = 0

    # horizontal_list: [ [x_min, x_max, y_min, y_max], ... ]
    if horizontal_list and horizontal_list[0]:
        for box in horizontal_list[0]:
            x_min, x_max, y_min, y_max = map(int, box)
            draw.rectangle([x_min, y_min, x_max, y_max], width=4)
            count += 1

    # free_list: 四点框
    if free_list and free_list[0]:
        for box in free_list[0]:
            pts = [(int(x), int(y)) for x, y in box]
            draw.line(pts + [pts[0]], width=4)
            count += 1

    img.save(save_path)
    return count


# In[17]:


candidate_dir = output_root / "text_detection_detectonly_preview"
candidate_dir.mkdir(parents=True, exist_ok=True)

good_candidates = []

for lm in LANDMARKS:
    imgs = [p for p in det_img_paths if get_label(p) == lm]
    print("\n类别:", lm, "图片数:", len(imgs))

    count_good = 0

    for img_path in imgs[:80]:
        save_path = candidate_dir / f"{lm}_{img_path.stem}_detectonly.jpg"
        box_count = draw_easyocr_detect_only(img_path, save_path)

        if box_count > 0:
            print("  有检测:", img_path.name, "框数:", box_count)
            good_candidates.append(img_path)
            count_good += 1

        if count_good >= 5:
            break

print("\n候选图数量:", len(good_candidates))


# In[18]:


selected = []

for lm in LANDMARKS:
    lm_good = [p for p in good_candidates if get_label(p) == lm]
    chosen = lm_good[:2]
    selected.extend(chosen)
    print(lm, "最终选择:", [p.name for p in chosen])

print("总计:", len(selected))
summary = []

for img_path in selected:
    lm = get_label(img_path)
    save_path = text_det_dir / f"{lm}_{img_path.stem}_detectonly.jpg"

    box_count = draw_easyocr_detect_only(img_path, save_path)

    summary.append({
        "landmark": lm,
        "image": img_path.name,
        "box_count": box_count,
        "output": str(save_path),
        "manual_eval": "待人工核验"
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(text_det_dir / "text_detection_summary.csv", index=False, encoding="utf-8-sig")

print("文字检测结果已保存到:", text_det_dir.resolve())
print("共生成:", len(summary_df), "张")
summary_df


# In[19]:


sample_det_paths = sorted(text_det_dir.glob("*.jpg"))[:5]

plt.figure(figsize=(15, 10))

for i, p in enumerate(sample_det_paths, 1):
    img = Image.open(p)
    plt.subplot(1, len(sample_det_paths), i)
    plt.imshow(img)
    plt.axis("off")
    plt.title(p.name[:20])

plt.tight_layout()
plt.show()


# In[20]:


print("=== 输出文件检查 ===")
print("P@K 曲线数量:", len(list(pk_curve_dir.glob("P@K_*.png"))))
print("文字检测可视化数量:", len(list(text_det_dir.glob("*.jpg"))))
print("P@K 结果表:", (pk_curve_dir / "retrieval_pk_results.csv").exists())
print("文字检测 summary:", (text_det_dir / "text_detection_summary.csv").exists())

print("\n项目输出目录:", output_root.resolve())

