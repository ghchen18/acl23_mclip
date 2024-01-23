#!/bin/bash

train_1stage_mte(){
    task='encfix_embfix_decrand'
    workloc='/path/to/your/work/location'

    modeldir=$workloc/models/mnmt_${task}_${MPLM}_m2m_m6  && mkdir -p $modeldir 
    fseq=$workloc/data/${DATASET}/fseq 
    dec_settings='--decoder-ffn-embed-dim 3072 --decoder-attention-heads 12 --decoder-layers 12 --decoder-embed-dim 768  --encoder-embed-dim 768  '

    lang_dict="en,cs,de,es,ja,ru,zh"
    lang_pairs="cs-en,de-en,es-en,ja-en,ru-en,zh-en,en-cs,en-de,en-es,en-ja,en-ru,en-zh"
    MaxUpdates=400000

    python train.py $fseq  --save-dir $modeldir  ${dec_settings} --seed 16 --fp16 --same-lang-per-batch --enable-lang-ids \
        --arch transformer --task translation_multi_simple_epoch --sampling-method 'temperature'  --sampling-temperature 5  \
        --langs ${lang_dict} --lang-pairs ${lang_pairs} --criterion label_smoothed_cross_entropy  \
        --label-smoothing 0.1  --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000  --max-update ${MaxUpdates} --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.0  --max-tokens 4096 --update-freq 1 --skip-invalid-size-inputs-valid-test  --log-interval 100 --truncate-source \
        --encoder-normalize-before --decoder-normalize-before --xlmr-task $task --tensorboard-logdir $modeldir/tensorboard \
        --share-all-embeddings --max-source-positions 512  --activation-fn gelu_accurate --xlmr-modeldir $workloc/models/${MPLM}_base \
        --log-format 'tqdm' --clip-norm 2.0 --num-workers 0 --save-interval-updates 8000 --MPLM-type ${MPLM} --ddp-backend=legacy_ddp \
        --validate-interval-updates 8000  --enable-lang-proj  \
        --optimizer adam --adam-betas '(0.9, 0.98)'  --lr 5e-4 \
    2>&1 | tee $modeldir/train.log

}


train_2stage_mte_ctl(){   
    task='xlmr_2stage'
    workloc='/path/to/your/work/location'

    modeldir=$workloc/models/mnmt_${task}_${MPLM}_m2m_ctl && mkdir -p $modeldir 
    fseq=$workloc/data/${DATASET}/fseq  

    dec_settings='--decoder-ffn-embed-dim 3072 --decoder-attention-heads 12 --decoder-layers 12 --decoder-embed-dim 768  --encoder-embed-dim 768  '

    lang_dict="en,cs,de,es,ja,ru,zh"
    lang_pairs="cs-en,de-en,es-en,ja-en,ru-en,zh-en,en-cs,en-de,en-es,en-ja,en-ru,en-zh"
    MaxUpdates=160000 

    python train.py $fseq --save-dir $modeldir  ${dec_settings} --seed 16 --fp16   \
        --arch transformer --task translation_multi_simple_epoch --sampling-method 'temperature'  --sampling-temperature 5  \
        --langs ${lang_dict} --lang-pairs ${lang_pairs} --criterion label_smoothed_cross_entropy_ctl --same-lang-per-batch --enable-lang-ids \
        --label-smoothing 0.1  --optimizer adam --adam-betas '(0.9, 0.98)'  --lr-scheduler inverse_sqrt  --lr 1e-4 --truncate-source  \
        --warmup-updates 10 --warmup-init-lr 0.00002  --max-update ${MaxUpdates} --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.0  --max-tokens 1024 --update-freq 4 --log-interval 100 --skip-invalid-size-inputs-valid-test  \
        --encoder-normalize-before --decoder-normalize-before --xlmr-task $task --tensorboard-logdir $modeldir/tensorboard \
        --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --activation-fn gelu_accurate --xlmr-modeldir $workloc/models/${MPLM}_base \
        --log-format 'tqdm' --clip-norm 2.0 --num-workers 0 --save-interval-updates 5000 --MPLM-type ${MPLM} \
        --load-from-which-ckpt $workloc/models/mnmt_encfix_embfix_decrand_${MPLM}_m2m_m6/checkpoint_best.pt  --ddp-backend=legacy_ddp  \
        --no-epoch-checkpoints --enable-lang-proj  --contrastive-lambda 2.0   \
    2>&1 | tee $modeldir/train.log  
}


extract_mte(){
    modeldir=$workloc/models/mnmt_xlmr_2stage_xlmr_ctl 
    fseq=$workloc/data/mnmt/fseq
    lang_dict="en,cs,de,es,ja,ru,zh"

    python scripts/test/extract_mte.py $fseq -s de -t en  \
        --path $modeldir/checkpoint_best.pt  --task translation \
        --max-tokens 5000  --langs ${lang_dict}  --MPLM-type 'xlmr'  --xlmr-task 'xlmr_2stage'

    cp $modeldir/mte_mclip.pt $workloc/models/xlmr_base/
}



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MPLM='xlmr'
export DATASET='mnmt'


echo "Train first stage of enhanced multilingual text encoder"
train_1stage_mte

echo "Train second stage of enhanced multilingual text encoder"
train_2stage_mte_ctl

echo "extract the encoder of NMT as enhanced MTE"
extract_mte

echo "finished"
