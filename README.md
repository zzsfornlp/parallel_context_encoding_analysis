## An Analysis of Parallel Context Encoding

Hi, this describes our implementation for our work of the analysis of parallel context encoding.

Please refer to the paper for more details: [[arxiv]](https://arxiv.org/abs/2412.16545) [[acl-anthology]](https://aclanthology.org/2025.acl-long.485/).

### Environment

Prepare the environment using conda:

    conda create -n s24p python=3.10
    pip install torch==2.4.0 transformers==4.44.0 pandas --extra-index-url https://download.pytorch.org/whl/cu118

Before running anything, make sure to export the src directory to your `$PYTHONPATH`:

    export PYTHONPATH=/your/path/to/src

### Data Preparation

#### Prepare LM Data

Use the script [get_lm_data.py](mspx/tasks/memana/scripts/get_lm_data.py) to prepare LM testing data, we use the `dev` subset for our experiments.

    python -m mspx.tasks.memana.scripts.get_lm_data pg19 pg19.dev.jsonl
    python -m mspx.tasks.memana.scripts.get_lm_data proof_pile proof_pile.dev.jsonl
    # shuffling data since we only take a subset to eval    
    shuf pg19.dev.jsonl >pg19.shuf.dev.jsonl
    shuf proof_pile.dev.jsonl >proof_pile.shuf.dev.jsonl

#### Prepare HELMET Data

Prepare the HELMET data (for RAG, ICL and Synthetic) by directly downloading preprocessed data from the HELMET repo: https://github.com/princeton-nlp/HELMET/blob/main/scripts/download_data.sh

Assuming that we put the decompressed data at the `data/` DIR.

### Running

#### Prepare Pretrained LM Checkpoints

Create a dir named `_cache` at your running dir and move the checkpoints of your pre-trained LM checkpoints there. (By default, we are setting the `cache_dir` of various huggingface loading to this dir.)

#### Evaluating LM

Here, we test LM with different types of parallel context settings:
- `test_context_split:${SP},${PPP},1024` means that we keep the final 1024 tokens for ppl eval, keep the middle `${PPP}` tokens unchanged, and then split the previous context into `${SP}` pieces. (For simplicity, we keep 1 middle token unsplit to easily calculate PPL.)
- `test_starting_prompt:p1` means adding a common prefix as the attention sink.
- `sel_topk:2` means enabling the selective mechanism (with topK=2).

```bash
# specifying the model
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B
# testing loop (testing for around 500K tokens)
function run_lm() {
for DATA_FILE in pg19.shuf.dev.jsonl proof_pile.shuf.dev.jsonl; do
if [[ ${DATA_FILE} =~ pg19 ]]; then
  ECOUNT=50
else
  ECOUNT=250
fi
for SS in 8192 4096 16384; do  # context length (4K 8K 16K)
for PPP in 1; do
for SP in 1 2 4 8 16 32 64 128; do  # parallel degree
for AAA in "" "test_starting_prompt:p1"; do  # attention sink
for BBB in "" "aggr_dims: sel_topk:2"; do  # selective attention
echo RUN with $DATA_FILE SS=$SS PPP=$PPP SP=$SP AAA=$AAA BBB=$BBB
ZCALC_ENTROPY=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m mspx.tasks.memana.run_ppl my_dtype:bf16 auto_device_map:1 my_device:0 model_name:mym:$MODEL_NAME input_file:$DATA_FILE inst_eval_count:$ECOUNT test_batch_size:1 test_seg_size:${SS} test_step_size:0 selatt.enabled:1 "test_context_split:${SP},${PPP},1024" $AAA $BBB
done; done; done; done; done
done |& tee _log_lm
}
# run them
run_lm()
```

#### Evaluating Other Tasks (RAG, ICL, Synthetic)

Here, we test RAG/ICL/Synthetic tasks with different settings. Setting are mostly similar to those in the LM eval.

```bash
# specifying the model
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B
# testing loop
function run_helmet() {
for INPUT_SIG in icl_8192 rag_8192 recall_8192 icl_4096 rag_4096 recall_4096 icl_16K rag_16K recall_16K; do
for SP in 1 2 4 8 16 32 64 128; do  # parallel degree
for AAA in "" "test_starting_prompt:p1"; do  # attention sink
for SA_EXTRA in "" "aggr_dims:head,seq sel_topk:2"; do  # selective attention
ZCALC_ENTROPY=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m mspx.tasks.memana.run_helmet my_dtype:bf16 my_device:0 model_name:mym:$MODEL_NAME input_sig:$INPUT_SIG test_context_split:$SP $AAA selatt.enabled:1 $SA_EXTRA "str_header:RUN with $INPUT_SIG SP=$SP AAA=$AAA SA_EXTRA=$SA_EXTRA"
done; done; done;
done |& tee _log_task
}
# run them
run_helmet()
```
