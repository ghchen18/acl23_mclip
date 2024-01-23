#!/bin/bash -e
import os
import pandas as pd
import argparse
import json,time 

def load_annotations(coco_dir):
    with open(os.path.join(coco_dir, 'annotations', f'captions_train2014.json')) as f:
        annotations = json.load(f)['annotations']

    with open(os.path.join(coco_dir, 'annotations', f'captions_val2014.json')) as f:
        annotations.extend(json.load(f)['annotations'])

    return annotations

def select_captions(annotations, image_ids, image_paths):
    image_ids = list(image_ids)
    image_Paths = list(image_paths)
    set_image_ids = set(image_ids)
    captions = []
    caption_image_ids = []
    caption_image_paths = []

    total = len(annotations)

    for idx, annotation in enumerate(annotations):
        if idx % 10000 ==0:
            ptime = time.strftime("%m-%d %H:%M", time.localtime())
            print(f"At {ptime} , now for line {idx} / {total} ...")

        image_id = annotation['image_id']
        if image_id in set_image_ids:
            indice = image_ids.index(image_id) 
            img_path = image_paths[indice]

            captions.append(annotation['caption'].replace('\n', ''))
            caption_image_ids.append(image_id)
            caption_image_paths.append(img_path)

    return captions, caption_image_ids, caption_image_paths

def write_captions(captions, filename, lowercase=False):
    with open(filename, 'w') as f:
        for caption in captions:
            if lowercase:
                caption = caption.lower()
            f.write(caption + '\n')

def write_image_ids(image_ids, filename):
    with open(filename, 'w') as f:
        for image_id in image_ids:
            f.write(f'{image_id}\n')

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations of MS-COCO training and validation set
    annotations = load_annotations(args.ms_coco_dir)

    split_df = pd.read_csv(f'{args.ms_coco_dir}/split/karpathy_{args.split}_images.txt', sep=' ', header=None)
    image_ids = split_df.iloc[:,1].to_numpy()
    image_paths = split_df.iloc[:,0].to_numpy()

    # Select captions and their image IDs from annotations
    captions, caption_image_ids, caption_image_paths = select_captions(annotations, image_ids, image_paths)

    captions_filename = os.path.join(args.output_dir, 'raw', f'{args.split}-captions.raw.en')
    caption_image_paths_filename = os.path.join(args.output_dir, 'fseq', f'{args.split}-ids.en.raw.txt')

    write_captions(captions, captions_filename)
    print(f'Wrote tokenized captions to {captions_filename}.')

    write_image_ids(caption_image_paths, caption_image_paths_filename)
    print(f'Wrote caption image paths to {caption_image_paths_filename}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')
    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test'],
                        help="Data split ('train', 'valid' or 'test').")
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')

    main(parser.parse_args())
