## Task:
I have a set of code '/workspace/scripts/finetune_lora.py''/workspace/scripts/infer_lora.py', '/workspace/scripts/utils.py' trying to do lora
  finetune on a sequential recommendation datasets. 
Your tasks are to review them and find out if any of my following concerns valid, and propose the change:

### My concerns:
1. I want you to examine the data preparison, I want you to make sure we have a test in /workspace/tests/test_utils.py, given the input, it converts it into the training input(basd on what template transformation we have), and verfy the results is in expected formats
2. we have a cut_off_len parameter, that I am afarid that we are choosing left or right cutoff incorrectly, that will affect the final prediction result
3. For example, the predicition from LLM is `yes, this is because xxx` or `</think>xxx</think>. yes,`, but the actual label is `yes`, will this be the case? if so, how can the loss being calculated correctly, can you review the way we do inference, and calculate loss. I am seeing after a fulfill tune, the model does not improved in accuracy. what can be happening in this case.
  
  
  ### Code to prepare the data: '/workspace/scripts/prepare_movielens.py'
  ### Test code: /workspace/tests/test_utils.py
  ### config: /workspace/configs/qwen3_7b_movielens_qlora.yaml

  ## Requirements:
  1. think very hard
  2. use some web search if needed for understanding how lora, loss, prediction task work to give better response
  3. don't make any change on code first, instead, write an investigation doc to answer all my concern with code refernece and good reasons. You can update the test case first to help with your evaluation.
