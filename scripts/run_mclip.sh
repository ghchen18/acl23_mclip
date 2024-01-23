#!/bin/bash
test_mclip(){
    modeldir=/path/to/your/checkpoint

    src='im' 
    lang_list="en,cs,de,fr,ja,zh,im"

	# coco
	langs=(en ja zh)
    for i in `seq 0 2`;do
    {
        tgt=${langs[$i]}
        type="coco_${tgt}"
        fseq=$workloc/data/testsets/$type/fseq  
        mscoco=$workloc/data/testsets/$type 

        CUDA_VISIBLE_DEVICES=$i  python scripts/test/eval_retrieval.py $fseq -s $src -t $tgt --path $modeldir/checkpoint_best.pt  \
            --batch-size 1024  --beam 5  --remove-bpe --fp16   --read-raw-images    \
            --mscoco-dir $mscoco --MPLM-type $MPLM  --enable-add-xlmr-bos --langs ${lang_list} \
            --model-overrides '{"bpe": "sentencepiece",  "sentencepiece_model":"'$workloc/models/xlmr_base/sentencepiece.bpe.model'", "save_dir":"'$modeldir'"}'  \
            --task translation_multi_simple_epoch 
        echo ">>> finish for xlmr coco-$tgt ... "
    }&
    done 
    wait 

	# multi30k
    langs=(en de fr cs)
    type="multi30k"  ## `seq 0 3`
    for i in `seq 0 3`;do
    {
        tgt=${langs[$i]}
        fseq=$workloc/data/testsets/multi30k/fseq 
        mscoco=$workloc/data/testsets/multi30k 

        CUDA_VISIBLE_DEVICES=$i  python scripts/test/eval_retrieval.py $fseq -s $src -t $tgt --path $modeldir/checkpoint_best.pt  \
            --batch-size 1024 --beam 5  --remove-bpe --fp16  --read-raw-images   \
            --mscoco-dir $mscoco --MPLM-type $MPLM  --enable-add-xlmr-bos --langs ${lang_list} \
            --model-overrides '{"bpe": "sentencepiece",  "sentencepiece_model":"'$workloc/models/xlmr_base/sentencepiece.bpe.model'", "save_dir":"'$modeldir'"}'  \
            --task translation_multi_simple_epoch  --arch 'multimodal_encoder_base'

        echo ">>> finish for multi30k $tgt ... "
    }&
    done 
    wait 
}


train_mclip(){   
    xtask='fixall_clinear_xlayer_mt6_ctl' 
    loss_type='sim'
    Mname="cc3m_${xtask}_${loss_type}_${MPLM}_release"  

    modeldir=$workloc/models/$Mname && mkdir -p $modeldir 
    fseq=$workloc/data/cc3m/fseq 
    mscoco=$workloc/data/cc3m
    lang_pairs='im-en' ## or  'im-en,im-ja,im-de,im-cs,im-fr'

    python train.py  $fseq \
        --task translation_multi_simple_epoch  -a 'multimodal_encoder_base' --optimizer lamb --lr 1e-2  --fp16-no-flatten-grads  \
        --label-smoothing 0.1 --dropout 0.3 --seed 16 --fp16  \
        --lr-scheduler inverse_sqrt --weight-decay 0.01 --criterion multimodal_ctl_loss --max-epoch 15 \
        --warmup-updates 500 --warmup-init-lr '1e-07' --keep-interval-updates 10 --no-epoch-checkpoints  \
        --save-dir $modeldir --ddp-backend=legacy_ddp --encoder-layers 2 \
        --max-target-positions 512  --log-format 'tqdm' \
        --clip-norm 2.0 --num-workers 2  --validate-interval-updates 250 \
        --log-interval 20 --save-interval-updates 250 --save-interval 2 \
        --sampling-method 'temperature'  --sampling-temperature 5 --loss-type ${loss_type}  \
        --batch-size 2048 --update-freq 1 --langs "en,cs,de,fr,ja,zh,im"  --lang-pairs=${lang_pairs} \
        --enable-add-xlmr-bos --mscoco-dir $mscoco  --MPLM-type $MPLM  --xlmr-task $xtask  \
        --tensorboard-logdir $modeldir/tensorboard   --truncate-target  --read-raw-images  \
        --bpe sentencepiece --sentencepiece-model $workloc/models/xlmr_base/sentencepiece.bpe.model \
    2>&1 | tee $modeldir/train.out 

}


ft_mclip_coco(){   
    DATASET='coco'  ## or 'multi30k'
    lang=$1 
    xtask='fixall_clinear_xlayer_mt6_ctl'
    loss_type='sim'
    Mname="coco_${xtask}_${loss_type}_${MPLM}_ft_with_${lang}_release"  

    modeldir=$workloc/models/$Mname && mkdir -p $modeldir 
    fseq=$workloc/data/${DATASET}/fseq  
    mscoco=$workloc/data/${DATASET}

    python train.py  $fseq \
        --task translation_multi_simple_epoch  -a 'multimodal_encoder_base'  \
        --optimizer lamb --lr 1e-2 --fp16-no-flatten-grads --fp16 \
        --label-smoothing 0.1 --dropout 0.3 --seed 16   \
        --lr-scheduler inverse_sqrt --weight-decay 0.001 --criterion 'multimodal_coco_ctl_loss' --max-epoch 30 \
        --warmup-updates 500 --warmup-init-lr '1e-07' --keep-last-epochs 5 --no-epoch-checkpoints  \
        --save-dir $modeldir --ddp-backend=legacy_ddp --encoder-layers 2 \
        --max-target-positions 512 --log-format 'tqdm' \
        --clip-norm 2.0 --num-workers 8  --validate-interval-updates 500 \
        --log-interval 20 --save-interval-updates 500 --save-interval 2 \
        --sampling-method 'temperature'  --sampling-temperature 5 --loss-type ${loss_type}  \
        --batch-size 128 --update-freq 1 --langs "en,cs,de,fr,ja,zh,im"  --lang-pairs "im-${lang}" \
        --enable-add-xlmr-bos --mscoco-dir $mscoco  --MPLM-type $MPLM  --xlmr-task $xtask --read-raw-images   \
        --bpe sentencepiece --sentencepiece-model $workloc/models/xlmr_base/sentencepiece.bpe.model \
        --load-from-which-ckpt $workloc/models/cc3m_fixall_clinear_xlayer_mt6_ctl_sim_xlmr/checkpoint_best.pt \
    2>&1 | tee  $modeldir/train.out   

}


export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
export DATASET=cc3m
export MPLM=xlmr


# echo "Train mCLIP with CC3M dataset using TriKD"
# train_mclip

echo "Test the zero-shot cross-modal retrieval performance "
test_mclip 

# echo "Fine-tune mCLIP with MSCOCO English dataset."
# ft_mclip_coco 'en'

echo "finished"
