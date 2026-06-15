
# coding: utf-8

# In[6]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# In[7]:


from pathlib import Path

DATA_ROOT = Path(r"E:\桌面\各种作业的东西\人工智能微专业\计算机视觉基础\交大视觉印象数据集2026\image_retrieval")

base_dir = DATA_ROOT / "base"
query_dir = DATA_ROOT / "query"

print(base_dir.exists(), query_dir.exists())

query_imgs = list(query_dir.glob("*.*"))
base_imgs = list(base_dir.rglob("*.*"))

print("query数量:", len(query_imgs))
print("base数量:", len(base_imgs))
print("前5个query:", [p.name for p in query_imgs[:5]])
print("前5个base:", [p.name for p in base_imgs[:5]])


# In[5]:


get_ipython().system('pip install tqdm')


# In[8]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np


# In[9]:


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

base_imgs = sorted([p for p in base_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
query_imgs = sorted([p for p in query_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

def get_label(path):
    return path.stem.split("-")[0]

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


# In[10]:


device = "cuda" if torch.cuda.is_available() else "cpu"

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

device


# In[12]:


@torch.no_grad()
def extract_features(paths, batch_size=64):
    ds = ImagePathDataset(paths, transform)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0)

    feats = []
    out_paths = []

    for imgs, batch_paths in dl:
        imgs = imgs.to(device)

        f = model(imgs)

        # L2归一化
        f = torch.nn.functional.normalize(f, dim=1)

        feats.append(f.cpu().numpy())
        out_paths.extend(batch_paths)

    feats = np.concatenate(feats, axis=0)

    return feats, [Path(p) for p in out_paths]


# In[13]:


base_feats, base_paths = extract_features(base_imgs, batch_size=64)
query_feats, query_paths = extract_features(query_imgs, batch_size=64)

print(base_feats.shape)
print(query_feats.shape)


# In[14]:


from PIL import Image, UnidentifiedImageError

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

print("有效base:", len(base_imgs))
print("坏base:", len(bad_base_imgs))
print("有效query:", len(query_imgs))
print("坏query:", len(bad_query_imgs))
print("坏图:", [p.name for p in bad_base_imgs[:10]])


# In[15]:


base_feats, base_paths = extract_features(base_imgs, batch_size=64)
query_feats, query_paths = extract_features(query_imgs, batch_size=64)

print(base_feats.shape)
print(query_feats.shape)


# In[16]:


similarity = query_feats @ base_feats.T
print(similarity.shape)


# In[17]:


idx = 0
top5 = similarity[idx].argsort()[-5:][::-1]

print("query:", query_paths[idx].name)
for rank, i in enumerate(top5, 1):
    print(rank, base_paths[i].name)


# In[18]:


LANDMARKS = ["fhy","jx", "kx","mh","nm","sjz","sy","tsg","ty","yf","yk","zx"]

def get_label(path):
    return path.stem.split("-")[0]

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

results


# In[20]:


from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print(output_dir.resolve())


# In[21]:


import matplotlib.pyplot as plt

Ks = [1, 5, 10, 20, 40, 60]

for lm in LANDMARKS:
    plt.figure(figsize=(4,3))
    plt.plot(Ks, results[lm], marker='o')
    plt.title(f"P@K - {lm}")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid(True)

    plt.savefig(output_dir / f"P@K_{lm}.png")

    plt.close()

print("完成，共保存", len(LANDMARKS), "张图")


# In[3]:


get_ipython().run_line_magic('pip', 'install paddleocr paddlepaddle')


# In[5]:


get_ipython().run_line_magic('pip', 'install -U typing_extensions')


# In[1]:


from paddleocr import PaddleOCR


# In[2]:


from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt

ocr = PaddleOCR(use_angle_cls=False, lang="ch")


# In[2]:


get_ipython().run_line_magic('pip', 'install easyocr')


# In[3]:


import easyocr
from PIL import Image, ImageDraw
from pathlib import Path

reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)


# In[9]:


from pathlib import Path

DATA_ROOT = Path(r"E:\桌面\各种作业的东西\人工智能微专业\计算机视觉基础\实验 阶段2\交大视觉印象数据集2026")

base_dir = DATA_ROOT / "image_retrieval" / "base"
query_dir = DATA_ROOT / "image_retrieval" / "query"
det_data_dir = DATA_ROOT / "object_detection" / "data"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

base_paths = sorted([p for p in base_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
query_paths = sorted([p for p in query_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
det_img_paths = sorted([p for p in det_data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

print("base:", len(base_paths))
print("query:", len(query_paths))
print("det_data:", len(det_img_paths))
print(base_paths[0])


# In[16]:


import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def draw_easyocr_boxes(img_path, save_path, conf_thres=0.3):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    result = reader.readtext(img_np)

    draw = ImageDraw.Draw(img)
    kept = []

    for box, text, conf in result:
        if conf < conf_thres:
            continue

        pts = [(int(x), int(y)) for x, y in box]
        draw.line(pts + [pts[0]], width=4)
        kept.append((box, text, conf))

    img.save(save_path)
    return kept, result


# In[17]:


preview_dir = Path("outputs/text_detection_preview")
preview_dir.mkdir(parents=True, exist_ok=True)

LANDMARKS = ["fhy","jx", "kx","mh","nm","sjz","sy","tsg","ty","yf","yk","zx"]

def get_label(path):
    return path.stem.split("-")[0]

good_candidates = []

for lm in LANDMARKS:
    imgs = [p for p in det_img_paths if get_label(p) == lm]
    print("\n类别:", lm, "图片数:", len(imgs))

    count = 0
    for img_path in imgs[:30]:  # 每类先试前30张
        save_path = preview_dir / f"{lm}_{img_path.stem}_easyocr.jpg"
        kept, raw = draw_easyocr_boxes(img_path, save_path, conf_thres=0.3)

        if len(kept) > 0:
            print("  有检测:", img_path.name, "有效框数:", len(kept))
            good_candidates.append(img_path)
            count += 1

        if count >= 3:
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


# In[19]:


final_dir = Path("outputs/text_detection_final")
final_dir.mkdir(parents=True, exist_ok=True)

summary = []

for img_path in selected:
    lm = get_label(img_path)
    save_path = final_dir / f"{lm}_{img_path.stem}_easyocr.jpg"

    kept, raw = draw_easyocr_boxes(img_path, save_path, conf_thres=0.3)

    summary.append({
        "landmark": lm,
        "image": img_path.name,
        "valid_boxes": len(kept),
        "raw_boxes": len(raw),
        "output": str(save_path)
    })

print("完成，保存到:", final_dir.resolve())
print("共生成:", len(summary), "张")


# In[20]:


import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

test_img = det_img_paths[0]
img = Image.open(test_img).convert("RGB")
img_np = np.array(img)

horizontal_list, free_list = reader.detect(img_np)

print("horizontal:", horizontal_list)
print("free:", free_list)


# In[21]:


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


# In[22]:


preview_dir = Path("outputs/text_detection_detectonly_preview")
preview_dir.mkdir(parents=True, exist_ok=True)

good_candidates = []

for lm in LANDMARKS:
    imgs = [p for p in det_img_paths if get_label(p) == lm]
    print("\n类别:", lm, "图片数:", len(imgs))

    count_good = 0
    for img_path in imgs[:80]:  
        save_path = preview_dir / f"{lm}_{img_path.stem}_detectonly.jpg"
        box_count = draw_easyocr_detect_only(img_path, save_path)

        if box_count > 0:
            print("  有检测:", img_path.name, "框数:", box_count)
            good_candidates.append(img_path)
            count_good += 1

        if count_good >= 5:
            break

print("候选图数量:", len(good_candidates))


# In[23]:


selected = []

for lm in LANDMARKS:
    lm_good = [p for p in good_candidates if get_label(p) == lm]
    chosen = lm_good[:2]
    selected.extend(chosen)
    print(lm, "最终选择:", [p.name for p in chosen])

print("总计:", len(selected))


# In[24]:


final_dir = Path("outputs/text_detection_detectonly_final")
final_dir.mkdir(parents=True, exist_ok=True)

summary = []

for img_path in selected:
    lm = get_label(img_path)
    save_path = final_dir / f"{lm}_{img_path.stem}_detectonly.jpg"

    box_count = draw_easyocr_detect_only(img_path, save_path)

    summary.append({
        "landmark": lm,
        "image": img_path.name,
        "box_count": box_count,
        "output": str(save_path)
    })

print("完成，保存到:", final_dir.resolve())
print("共生成:", len(summary), "张")

