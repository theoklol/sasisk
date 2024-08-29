import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import argparse
import os

import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args

import Models

MODEL_REPO_MAP = {
    "eva02_large": "SmilingWolf/wd-eva02-large-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit_large": "SmilingWolf/wd-vit-large-tagger-v3",
    "swinv2_v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    def jit_predict(self, x):
        # Not actually JITed since this is a single shot script,
        # but this is the function you would decorate with @jax.jit
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    # Resize
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.BICUBIC,
        )
    return image

'''def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]                         # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image'''

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

'''class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)'''
  
def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def load_model_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> PredModel:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.msgpack",
        revision=revision,
        token=token,
    )

    model_config = hf_hub_download(
        repo_id=repo_id,
        filename="sw_jax_cv_config.json",
        revision=revision,
        token=token,
    )

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]

def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Any,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(
            gen_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(
            char_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)


def main(args):
    # 画像を読み込む
    repo_id = MODEL_REPO_MAP.get(args.model_dir)
    train_data_dir = Path(args.train_data_dir)
    #images_path = train_util.glob_images_pathlib(train_data_dir, args.recursive)
    #print(f"found {len(images_path)} images.")

    print(f"Loading model '{args.model_dir}' from '{repo_id}'...")
    model, target_size  = load_model_hf(repo_id=repo_id)
    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Loading image and preprocessing...", train_data_dir)
    # get image

    for file in train_data_dir.rglob('*'):
        print("here.")
        if file.is_file():
            print(file)
            img_input: Image.Image = Image.open(file)

            img_input = pil_ensure_rgb(img_input)
            # pad to square with white background
            img_input = pil_pad_square(img_input)
            img_input = pil_resize(img_input, target_size)
            # convert to numpy array and add batch dimension
            inputs = np.array(img_input)
            inputs = np.expand_dims(inputs, axis=0)
            # NHWC image RGB to BGR
            inputs = inputs[..., ::-1]

            print("Running inference...")
            outputs = model.predict(inputs)

            print("Processing results...")
            caption, taglist, ratings, character, general = get_tags(
                probs=outputs,
                labels=labels,
                gen_threshold=args.general_threshold,
                char_threshold=args.character_threshold,
            )


            combined_tags = []
            general_tag_text = ""
            character_tag_text = ""

            for k, v in character.items():
                combined_tags.append(k)

            for k, v in general.items():
                combined_tags.append(k)

            tag_text = ', '.join(combined_tags)
                        
            with open(os.path.splitext(file)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
                f.write(tag_text + '\n')
                if args.debug:
                    print(f"\n{file}:\n  tags: {tag_text}")




    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    # 依存ライブラリを増やしたくないので自力で読むよ



    '''with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    l = [row for row in reader]
                    header = l[0]             # tag_id,name,category,count
                    rows = l[1:]
                assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"unexpected csv format: {header}"
            
                general_tags = [row[1] for row in rows[1:] if row[2] == '0']
                character_tags = [row[1] for row in rows[1:] if row[2] == '4']
            
                # 画像を読み込む
                
                train_data_dir = Path(args.train_data_dir)
                image_paths = train_util.glob_images_pathlib(train_data_dir, args.recursive)
                print(f"found {len(image_paths)} images.")
            
                tag_freq = {}
            
                undesired_tags = set(args.undesired_tags.split(','))
            
                def run_batch(path_imgs):
                    imgs = np.array([im for _, im in path_imgs])
            
                    probs = model(imgs, training=False)
                    probs = probs.numpy()
            
                    for (image_path, _), prob in zip(path_imgs, probs):
                        # 最初の4つはratingなので無視する
                        # # First 4 labels are actually ratings: pick one with argmax
                        # ratings_names = label_names[:4]
                        # rating_index = ratings_names["probs"].argmax()
                        # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]
            
                        # それ以降はタグなのでconfidenceがthresholdより高いものを追加する
                        # Everything else is tags: pick any where prediction confidence > threshold
                        combined_tags = []
                        general_tag_text = ""
                        character_tag_text = ""
                        for i, p in enumerate(prob[4:]):
                            if i < len(general_tags) and p >= args.general_threshold:
                                tag_name = general_tags[i].replace('_', ' ') if args.remove_underscore else general_tags[i]
                                if tag_name not in undesired_tags:
                                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                                    general_tag_text += ", " + tag_name
                                    combined_tags.append(tag_name)
                            elif i >= len(general_tags) and p >= args.character_threshold:
                                tag_name = character_tags[i - len(general_tags)].replace('_', ' ') if args.remove_underscore else character_tags[i - len(general_tags)]
                                if tag_name not in undesired_tags:
                                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                                    character_tag_text += ", " + tag_name
                                    combined_tags.append(tag_name)
            
                        if len(general_tag_text) > 0:
                            general_tag_text = general_tag_text[2:]
            
                        if len(character_tag_text) > 0:
                            character_tag_text = character_tag_text[2:]
            
                        tag_text = ', '.join(combined_tags)
                        
                        with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
                            f.write(tag_text + '\n')
                            if args.debug:
                                print(f"\n{image_path}:\n  Character tags: {character_tag_text}\n  General tags: {general_tag_text}")
            
            
                # 読み込みの高速化のためにDataLoaderを使うオプション
                if args.max_data_loader_n_workers is not None:
                    dataset = ImageLoadingPrepDataset(image_paths)
                    data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
                else:
                    data = [[(None, ip)] for ip in image_paths]
            
                b_imgs = []
                for data_entry in tqdm(data, smoothing=0.0):
                    for data in data_entry:
                        if data is None:
                            continue
            
                        image, image_path = data
                        if image is not None:
                            image = image.detach().numpy()
                        else:
                            try:
                                image = Image.open(image_path)
                                if image.mode != 'RGB':
                                    image = image.convert("RGB")
                                image = preprocess_image(image)
                            except Exception as e:
                                print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                                continue
                        b_imgs.append((image_path, image))
            
                        if len(b_imgs) >= args.batch_size:
                            b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                            run_batch(b_imgs)
                            b_imgs.clear()
            
                if len(b_imgs) > 0:
                    b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                    run_batch(b_imgs)
            
                if args.frequency_tags:
                    sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
                    print("\nTag frequencies:")
                    for tag, freq in sorted_tags:
                        print(f"{tag}: {freq}")'''

    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--repo_id", type=str, default=MODEL_REPO_MAP["convnext"],
    help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model",
    help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ")
    parser.add_argument("--force_download", action='store_true',
    help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
    help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--caption_extention", type=str, default=None,
    help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--general_threshold", type=float, default=0.35, help="threshold of confidence to add a tag for general category")
    parser.add_argument("--character_threshold", type=float, default=0.35, help="threshold of confidence to add a tag for character category")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively")
    parser.add_argument("--remove_underscore", action="store_true", help="replace underscores with spaces in the output tags")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--undesired_tags", type=str, default="", help="comma-separated list of undesired tags to remove from the output")
    parser.add_argument('--frequency_tags', action='store_true', help='Show frequency of tags for images')

    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)
