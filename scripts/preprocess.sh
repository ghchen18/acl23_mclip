#!/bin/bash

binarize_machine_translation_dataset(){
    tgt=en
    workloc='/path/to/your/work/location'

    raw=$workloc/data/mnmt/raw 
    bpe=$workloc/data/mnmt/bpe 
    fseq=$workloc/data/mnmt/fseq
    mkdir -p $raw $bpe $fseq 

    for src in 'de' 'es' 'cs' 'ja' 'ru' 'zh'; do 
        for f in "valid.$src-en.$src" "valid.$src-en.en" "train.$src-en.$src" "train.$src-en.en"; do 
            python scripts/process/spm_encode.py --model $workloc/models/xlmr_base/sentencepiece.bpe.model  \
                --inputs $raw/$f --outputs $bpe/$f
        done 

        python fairseq_cli/preprocess.py -s $src -t $tgt --dataset-impl lazy \
            --workers 24 --destdir $fseq --validpref $bpe/valid.$src-en   \
            --trainpref $bpe/train.$src-en \
            --srcdict  $workloc/models/xlmr_base/dict.txt \
            --tgtdict $workloc/models/xlmr_base/dict.txt 
    done 
}


prepare_mscoco(){
    # Download [MSCOCO 2014 data](https://cocodataset.org/#download) and [Kapathy split](https://github.com/karpathy/neuraltalk2/issues/192)

    # After processing, the path of MSCOCO is like following:
    # --coco2014
    #     --train2014 ## all train2014 images
    #     --val2014 ## all val2014 images
    #     --fseq
    #         * Binarized fairseq files 
    #         * {train,valid,test}-ids.{lang}.raw.txt # each line is the image path of the corresponding caption under mscoco-path='coco2014'
    #     --split ## kapathy split files

    root='/path/to/your/work/location'
    workloc=$root/data/coco 
    fseq=$workloc/fseq && bpe=$fseq/bpe && raw=$fseq/raw && mkdir -p $bpe $raw 
    
    echo "Now use kapathy split to create MSCOCO data."
    for split in 'valid' 'train' 'test'; do
        python scripts/process/process_coco_captions.py --output-dir $workloc --ms-coco-dir $workloc --split $split
    done 

    echo "Extract and process the image captions"
    for split in 'valid' 'train' 'test'; do 
        python scripts/process/spm_encode.py --model $root/models/xlmr_base/sentencepiece.bpe.model  \
            --inputs $raw/$split-captions.raw.en --outputs $bpe/$split-captions.spm.en
    done 

    python fairseq_cli/preprocess.py -s 'en' --only-source --dataset-impl lazy \
        --workers 8 --destdir $fseq --trainpref $bpe/train-captions.spm   \
        --validpref $bpe/valid-captions.spm  \
        --srcdict $root/models/xlmr_base/dict.txt 

    cp $fseq/dict.en.txt $fseq/dict.im.txt ## view image as a special type of language 'im'

}

prepare_cc3m(){
    workloc=$root/data/cc3m 
    split='train'

    fseq=$workloc/fseq && bpe=$fseq/bpe && raw=$fseq/raw && mkdir -p $bpe $raw 
    perl scripts/process/detokenizer.perl < $raw/$split-captions.raw.en > $raw/$split-captions.raw.detok.en

    python scripts/process/spm_encode.py --model $root/models/xlmr_base/sentencepiece.bpe.model  \
        --inputs $raw/$split-captions.raw.detok.en --outputs $bpe/$split-captions.spm.en

    python fairseq_cli/preprocess.py -s 'en' --only-source --dataset-impl lazy \
        --workers 8 --destdir $fseq --trainpref $bpe/train-captions.spm  \
        --srcdict $root/models/xlmr_base/dict.txt 
    
    cp $fseq/dict.en.txt $fseq/dict.im.txt ## view image as a special type of language 'im'
    
    # Build the image path files ({train,valid}-ids.{lang}.raw.txt) similarily. The valid set is from XTD10. 
}


prepare_multi30k(){
    # Download multi30k data from <https://github.com/multi30k/dataset>. 
    # For En and De, use the train data in task2, For Cs and Fr, use the train data in task1. 
    # Use test_2016 in task 1 as test set. 

    workloc=$root/data/multi30k  && mkdir -p $workloc/raw 
    raw=$workloc/fseq/raw && mkdir -p $raw
    fseq=$workloc/fseq  
    bpe=$fseq/bpe && mkdir -p $bpe $fseq 
    mkdir -p $workloc/images_train $workloc/images_valid
    
    for src in 'en' 'de' 'cs' 'fr'; do
        for split in 'train' 'valid' 'test'; do 
            perl scripts/process/detokenizer.perl < $raw/${split}.tok.${src} > $raw/${split}.${src}
            python scripts/process/spm_encode.py --model $root/models/xlmr_base/sentencepiece.bpe.model  \
                --inputs $raw/$split.$src --outputs $bpe/$split.spm.$src 
        done 

        python fairseq_cli/preprocess.py -s $src --only-source --dataset-impl lazy \
            --workers 8 --destdir $fseq  --validpref $bpe/valid.spm \
            --trainpref $bpe/train.spm --testpref $bpe/test.spm \
            --srcdict $root/models/xlmr_base/dict.txt 
    done 
    cp $fseq/dict.en.txt $fseq/dict.im.txt ## view image as a special type of language 'im' 

}


binarize_machine_translation_dataset

prepare_cc3m

prepare_mscoco

prepare_multi30k


