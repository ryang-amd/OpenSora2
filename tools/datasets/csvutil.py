import argparse
import csv
import html
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def get_video_length(path):
    import cv2

    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


LLAVA_PREFIX = [
    "The video shows",
    "The video captures",
    "The video features",
    "The video depicts",
    "The video presents",
    "The video features",
    "The video is ",
    "In the video,",
]


def remove_caption_prefix(caption):
    for prefix in LLAVA_PREFIX:
        if caption.startswith(prefix):
            caption = caption[len(prefix) :].strip()
            if caption[0].islower():
                caption = caption[0].upper() + caption[1:]
            return caption


def build_lang_detector(lang_to_detect):
    from lingua import Language, LanguageDetectorBuilder

    lang_dict = dict(en=Language.ENGLISH)
    assert lang_to_detect in lang_dict
    valid_lang = lang_dict[lang_to_detect]
    detector = LanguageDetectorBuilder.from_all_spoken_languages().with_low_accuracy_mode().build()

    def detect_lang(caption):
        confidence_values = detector.compute_language_confidence_values(caption)
        confidence = [x.language for x in confidence_values[:5]]
        if valid_lang not in confidence:
            return False
        return True

    return detect_lang


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+")
    parser.add_argument("--output", type=str, default=None)
    # special case
    parser.add_argument("--shard", type=int, default=None)

    # path processing
    parser.add_argument("--abspath", type=str, default=None)
    parser.add_argument("--relpath", type=str, default=None)
    # caption filtering
    parser.add_argument("--remove-empty-caption", action="store_true")
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--remove-url", action="store_true")
    # caption processing
    parser.add_argument("--remove-caption-prefix", action="store_true")
    parser.add_argument("--unescape", action="store_true")
    # num_frames processing
    parser.add_argument("--relength", action="store_true")
    # num_frames filtering
    parser.add_argument("--fmin", type=int, default=None)
    parser.add_argument("--fmax", type=int, default=None)
    # aesthetic filtering
    parser.add_argument("--aesmin", type=float, default=None)

    return parser.parse_args()


def get_output_path(args, input_name):
    if args.output is not None:
        return args.output

    name = input_name
    dir_path = os.path.dirname(args.input[0])

    # path processing
    if args.abspath is not None:
        name += "_abspath"
    if args.relpath is not None:
        name += "_relpath"
    # caption filtering
    if args.remove_empty_caption:
        name += "_rec"
    if args.lang is not None:
        name += f"_{args.lang}"
    if args.remove_url:
        name += "_nourl"
    # caption processing
    if args.remove_caption_prefix:
        name += "_rcp"
    if args.unescape:
        name += "_unescape"
    # num_frames processing
    if args.relength:
        name += "_relength"
    # num_frames filtering
    if args.fmin is not None:
        name += f"_fmin_{args.fmin}"
    if args.fmax is not None:
        name += f"_fmax_{args.fmax}"
    # aesthetic filtering
    if args.aesmin is not None:
        name += f"_aesmin_{args.aesmin}"

    output_path = os.path.join(dir_path, f"{name}.csv")
    return output_path


def main(args):
    # reading data
    data = []
    input_name = ""
    for i, input_path in enumerate(args.input):
        data.append(pd.read_csv(input_path))
        input_name += os.path.basename(input_path).split(".")[0]
        if i != len(args.input) - 1:
            input_name += "|"
        print(f"Loaded {len(data[-1])} samples from {input_path}.")
    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"Total number of samples: {len(data)}.")

    # get output path
    output_path = get_output_path(args, input_name)

    # preparation
    if args.lang is not None:
        detect_lang = build_lang_detector(args.lang)

    # processing
    if args.abspath is not None:
        assert args.relpath is None
        data["path"] = data["path"].progress_apply(lambda x: os.path.join(args.abspath, x))
    if args.relpath is not None:
        assert args.abspath is None
        data["path"] = data["path"].progress_apply(lambda x: os.path.relpath(x, args.relpath))
    if args.remove_caption_prefix:
        assert "text" in data.columns
        data["text"] = data["text"].progress_apply(remove_caption_prefix)
    if args.unescape:
        assert "text" in data.columns
        data["text"] = data["text"].progress_apply(html.unescape)
    if args.relength:
        data["num_frames"] = data["path"].progress_apply(get_video_length)

    # filtering
    if args.remove_empty_caption:
        assert "text" in data.columns
        data = data[data["text"].str.len() > 0]
    if args.remove_url:
        assert "text" in data.columns
        data = data[~data["text"].str.contains(r"(?P<url>https?://[^\s]+)", regex=True)]
    if args.lang is not None:
        assert "text" in data.columns
        data = data[data["text"].progress_apply(detect_lang)]
    if args.fmin is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] >= args.fmin]
    if args.fmax is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] <= args.fmax]
    if args.aesmin is not None:
        assert "aesthetic_score" in data.columns
        data = data[data["aesthetic_score"] >= args.aesmin]
    print(f"Filtered number of samples: {len(data)}.")

    # shard data
    if args.shard is not None:
        sharded_data = np.array_split(data, args.shard)
        for i in range(args.shard):
            output_path = output_path.replace(".csv", f"_{i}.csv")
            sharded_data[i].to_csv(output_path, index=False)
            print(f"Saved {len(sharded_data[i])} samples to {output_path}.")
    else:
        data.to_csv(output_path, index=False)
        print(f"Saved {len(data)} samples to {output_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
