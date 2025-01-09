# granite3.1-gradio
2 liners gradio chatbot with llamacpp and granite-3.1-2b-instruct-GGUF

<img src='https://newsroom.ibm.com/image/Granite_banner+%281%29.jpg' wodth=900>


Create a fast chatbot with 2 lines of code.

This example is good only for quick check, but it doesn't allow much control


### Folder structure
```
granite3.1-gradio
├───llamacpp
│   └───model
└───venv
```


### INSTRUCTIONS
Clone the Repo to have already the directory structure and the llamaCPP binaries
- the model (GGUF) goes inside the subfolder `llamacpp/model`
- extract the content of the ZIP archive `llama-b4435-bin-win-vulkan-x64.zip` inside the `llamacpp` directory
- in tha main project folder create a virtual environment
- install the following dependencies:
  `pip install --upgrade gradio tiktoken openai`

In one terminal window, from the `llamacpp` directory start the llama-server:
```bash
llama-server.exe -m model\granite-3.1-2b-instruct-Q5_K_L.gguf -c 8192 -ngl 0 --temp 0.2 --repeat-penalty 1.45
```

with the venv activated then run:
```bash
python .\testgradioGRANITE.py
```

####
Used stack:
- python 3.12
- llamacpp
- granite-3.1-2b-instruct-Q5_K_L.gguf
- gradio

### MODEL CARD
```
srv    load_model: loading model '.\model\granite-3.1-2b-instruct-Q5_K_L.gguf'
llama_model_loader: loaded meta data with 44 key-value pairs and 362 tensors from .\model\granite-3.1-2b-instruct-Q5_K_L.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
        general.architecture str              = granite
                general.type str              = model
                general.name str              = Granite 3.1 2b Instruct
            general.finetune str              = instruct
            general.basename str              = granite-3.1
          general.size_label str              = 2B
             general.license str              = apache-2.0
    general.base_model.count u32              = 1
   general.base_model.0.name str              = Granite 3.1 2b Base
al.base_model.0.organization str              = Ibm Granite
eneral.base_model.0.repo_url str              = https://huggingface.co/ibm-granite/gr...
                general.tags arr[str,3]       = ["language", "granite-3.1", "text-gen...
         granite.block_count u32              = 40
      granite.context_length u32              = 131072
    granite.embedding_length u32              = 2048
 granite.feed_forward_length u32              = 8192
granite.attention.head_count u32              = 32
nite.attention.head_count_kv u32              = 8
      granite.rope.freq_base f32              = 5000000.000000
ntion.layer_norm_rms_epsilon f32              = 0.000010
           general.file_type u32              = 17
          granite.vocab_size u32              = 49155
granite.rope.dimension_count u32              = 64
     granite.attention.scale f32              = 0.015625
     granite.embedding_scale f32              = 12.000000
      granite.residual_scale f32              = 0.220000
         granite.logit_scale f32              = 8.000000
        tokenizer.ggml.model str              = gpt2
          tokenizer.ggml.pre str              = refact
       tokenizer.ggml.tokens arr[str,49155]   = ["<|end_of_text|>", "<fim_prefix>", "...
   tokenizer.ggml.token_type arr[i32,49155]   = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
       tokenizer.ggml.merges arr[str,48891]   = ["─á ─á", "─á─á ─á─á", "─á─á─á─á ─á─á...
 tokenizer.ggml.bos_token_id u32              = 0
 tokenizer.ggml.eos_token_id u32              = 0
enizer.ggml.unknown_token_id u32              = 0
enizer.ggml.padding_token_id u32              = 0
tokenizer.ggml.add_bos_token bool             = false
     tokenizer.chat_template str              = {%- if messages[0]['role'] == 'system...
enizer.ggml.add_space_prefix bool             = false
general.quantization_version u32              = 2
       quantize.imatrix.file str              = /models_out/granite-3.1-2b-instruct-G...
    quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
antize.imatrix.entries_count i32              = 280
uantize.imatrix.chunks_count i32              = 152
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q5_K:  240 tensors
llama_model_loader: - type q6_K:   40 tensors
llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
llm_load_vocab: special tokens cache size = 22
llm_load_vocab: token to piece cache size = 0.2826 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = granite
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 49155
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 2.53 B
llm_load_print_meta: model size       = 1.70 GiB (5.77 BPW)
llm_load_print_meta: general.name     = Granite 3.1 2b Instruct
llm_load_print_meta: BOS token        = 0 '<|end_of_text|>'
llm_load_print_meta: EOS token        = 0 '<|end_of_text|>'
llm_load_print_meta: UNK token        = 0 '<|end_of_text|>'
llm_load_print_meta: PAD token        = 0 '<|end_of_text|>'
llm_load_print_meta: LF token         = 145 '├ä'
llm_load_print_meta: EOG token        = 0 '<|end_of_text|>'
llm_load_print_meta: max token length = 512
llm_load_print_meta: f_embedding_scale = 12.000000
llm_load_print_meta: f_residual_scale  = 0.220000
llm_load_print_meta: f_attention_scale = 0.015625
ggml_vulkan: Compiling shaders...................................................Done!
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/41 layers to GPU
llm_load_tensors:   CPU_Mapped model buffer size =  1742.80 MiB
................................................................................................
llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 8192
llama_new_context_with_model: n_ctx_per_seq = 8192
llama_new_context_with_model: n_batch       = 2048
llama_new_context_with_model: n_ubatch      = 512
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 5000000.0
llama_new_context_with_model: freq_scale    = 1
llama_new_context_with_model: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 8192, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 40, can_shift = 1
llama_kv_cache_init:        CPU KV buffer size =   640.00 MiB
llama_new_context_with_model: KV self size  =  640.00 MiB, K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.19 MiB
llama_new_context_with_model:    Vulkan0 compute buffer size =   562.75 MiB
llama_new_context_with_model: Vulkan_Host compute buffer size =    20.01 MiB
llama_new_context_with_model: graph nodes  = 1368
llama_new_context_with_model: graph splits = 444 (with bs=512), 1 (with bs=1)
common_init_from_params: setting dry_penalty_last_n to ctx_size = 8192
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 1
slot         init: id  0 | task -1 | new slot n_ctx_slot = 8192
main: model loaded
main: chat template, chat_template: (built-in), example_format: '<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Hi there<|end_of_text|>
<|start_of_role|>user<|end_of_role|>How are you?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```
