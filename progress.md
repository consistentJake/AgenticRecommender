## Jan 12

new module structure 
```

- [ ] for these tests file, can we centralized the place we set up API key
- [ ] in outputs/stage2_enriched_users.json, I think we should add more meta data into cuisine_sequence
```
    "cuisine_sequence": [
      "chinese",
      "pizza",
      "thai",
      "thai",
      "thai",
      "cakes",
      "fried chicken",
      "burgers",
      "bakery"
    ],
```s


## Jan 13 improvement plan
the workflow stars from do `agentic_recommender/workflow/workflow_runner.py --config agentic_recommender/workflow/workflow_config_linux.yaml`. you will find the logic and code starting from here.
Followings are several points I want to make updates, please complete them one by one but think about how to update the code in a systematic way.
1. in preparing the enriched_user data, I want to make change of how we record the history purchase behavior, we should not just record the list of cuisine, we should also for each cuisine, we record the vendor id, the hour, the week day the purchase happen. 
2. therefore during prediction task, we want to predict the next item after the user's history, the hour and the week day should be given, and then we predict the cuisine. 
3. when we are storing the processed data, I like the way we create preview like outputs/stage1_merged_preview.json, so that i can talk a look of the data generated rather than we have so many big file and hard to look at. so keep the full data json file is okay, but also add a step to create a small preview file for each json file we generated.



## Jan 19 improvement plan
Help me design the update plan of the current code base
### task
1. instead of we truncate the purchase history, we use all the purchase history. for example a user has purchase history of 50 items, we use all 49 items as inputs, and the task is trying to recommend the last item.
2. set up a user filter, we now have a minimal order history count, let's also use a maximum orer history count, let's use inf for now. Therefore, for a user that have more items than the max order history count, those user will get filter out.
3. how we generate the top 20 candidates pool. We use swing similiarty to calculate item-item similiarty. Then for the user's last n - 1 items, first we do dedup, have a unqie list of items. For each item, then we use swing similiarty to get the top k item simialr to this item. we then have (n - 1) * k items, now we need to dedup again and sort them by the swing similiarty score. 
4. gievn the top 20 candiates, and the existing user profiles, with all the first n - 1 purchased items, we now let the llm instead of predict the next items, we ask them to re-rank the top 20 candiates.
5. we have a reflection round, (let's call it critics), based on the response from the llm, we want the system to justify it. We add a new inforamtion, which is a user-item simialrity score calcualted by lightGCN. So lightGCN will have each user and item a embedding vector, for the top 20 candiates, we calculate the lightGCN based similiarty score between the user and the 20 items, provid the simialirty score, also a lightGCN based re-rank 20 items. 
6. in the reflection round, we combine both response from first LLM recommendation round: user profile with n - 1 item history, first round re-rank 20 items, the new lightGNC based re-ranked 20 items and thier lightGCN based score. -> ask the llm to re-rank the candiates again. get a final re-ranked list
7. use the re-ranking list, we can calulate the metrics. NDCG@K	Quality of ranking positions	Weighted sum of relevance by log discount, normalized by ideal
MRR@K	How soon the first correct item appears	Average reciprocal of rank positions
HitRate@K	Whether the correct item is in top-K	Fraction of lists containing the true item

6. 
## requirements and instruction
1. starting point is agentic_recommender/workflow/workflow_runner.py, any changes are related to the codd used in workflow_runner.py 
2. how we calculate with lightGCN? so I provide a example code to calcualte lightGCN.  agentic_recommender/similarity/lightGCN.py. This is also something we need to pre-calulated, once the user and item relationships is generated, we calcualte the lightGCN and get the embedding presenatation for each user and item, and cache them. Because we use the same dataset(dataset name is data_se, or data_sg, or other, just cache them with the original name), once we calcualted, in next adn future run we don;'t need to recalcuate just read from the cache. 
3. how we calculate swing similarity? you can refer and use this agentic_recommender/similarity/methods.py

## Jan 22
round2_prompt, let's include the timestamp.
check lightGCN training, do we include the prediction into training
  "round1_hit@5": 1.0,
  "final_hit@5": 1.0,
  include round1 hit@1

use evaulation data instead of training data.

rerun the model mentioned in https://arxiv.org/pdf/2402.01339
if we can't, we shall use lightGCN.


## Jan 25

 1. keep two methods: one is training with leave-last-out training data with lightGCN, and input is the     
  previosu items, do prediction in last item. let's call it PureTrainingData method.therefore the only code  
  change you need is to update the data loading and how we train the lightGCN.                               
  2. we will have method 2. train with full lightGCN data, but we are using full history data from training  
  as the input, and then use the test data as the last item. \                                               
  3. in test data, one customer have multiple orders, each order can have one or more items, different       
  order have different order time. Therefore, when we construct the testing data, we use full history from   
  training as the input, and create one test case based on one order in the testing data, therefore, one     
  user in testing data can have x test case, x is the order number in test data. \                           
  4. when we are doing evaluation, let consider this carefully. I create a doc                               
  '/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/evaluation/basket_evaluation.md'   
  to handle this case, which is the basket prediction case. think of how to revise the code in               
  '/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/evaluation', therefore we make     
  the calcualtion sitaution for a one item in test case, and multiple item of one order in test case.\       
  5.    1. keep two methods: one is training with leave-last-out training data with lightGCN, and input is the     
  previosu items, do prediction in last item. let's call it PureTrainingData method.therefore the only code  
  change you need is to update the data loading and how we train the lightGCN.                               
  2. we will have method 2. train with full lightGCN data, but we are using full history data from training  
  as the input, and then use the test data as the last item. \                                               
  3. in test data, one customer have multiple orders, each order can have one or more items, different       
  order have different order time. Therefore, when we construct the testing data, we use full history from   
  training as the input, and create one test case based on one order in the testing data, therefore, one     
  user in testing data can have x test case, x is the order number in test data. \                           
  4. when we are doing evaluation, let consider this carefully. I create a doc                               
  '/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/evaluation/basket_evaluation.md'   
  to handle this case, which is the basket prediction case. think of how to revise the code in               
  '/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/evaluation', therefore we make     
  the calcualtion sitaution for a one item in test case, and multiple item of one order in test case.\       
  



  ### jan 30

  I am looking at my last run with '/home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/workflow/workflow_config_qwen32_linux.yaml'. the result is in  
  '/home/zhenkai/personal/Projects/AgenticRecommender/outputs/202601262250/stage8_enhanced_rerank_detailed.json'. several things I need your investigation and fix:\      
  1. in this detailed logs, I need you to also log the request we send to llm model. and in a good formated way of logging, with nice indent. currently the request of    
  first round and second round is missing. 2. in the                                                                                                                      
  result,'/home/zhenkai/personal/Projects/AgenticRecommender/outputs/202601262250/stage8_enhanced_rerank_detailed.json', why the historical items are not                 
  `vendor_id||cuisine||(weekday, hour)` format, and why it seems like we have dedup the list of items too. I think as long as we select the `        prediction_target:   
  "vendor_cuisine"` in config, we should follow this selection all the ways in each stage. please verify if the code is doing the right thing. 3. I know I want to reuse  
  the calculation results for previous round of running, so I think we reuse some results of the stages before stage 8, are we resuse those results are saved in          
  '/home/zhenkai/personal/Projects/AgenticRecommender/outputs'. Can we make sure, running the workflow_runner.py we can specify that we want to regenerate all the        
  previous cached results. carefully think about that, is the swing similarty is affected, is the lightGCN is affected, then we have to also give a command to            
  recalculate those too. but generally, the lightGCN and swing thing, if we are assume we don't change, (i believe if we choose the same `prediction_target`, these two   
  calculation can not be changed`. be aware that, we always in swing and lightGCN, we only do training and calculation with `predication_target` for example              
  vendor_id||cuisine, but in LLM prompt, we must make sure we give the LLM `vendor_id||cuisine||(weekday, hour)`. so, think of how to optimize the user profile data      
  structure we create, that we can easily pull out the information we want for lightGCN/swing clculation (most likely it is user:item -> frequence) and information for   
  the final LLM request, user: full historic recores( item + time tuples)  


  ## Jan 31

## Improvement task
  1. In the following, we only consider `vendor_id||primary_cuisine` as the `item`, therefore one order is one single datapoints, even one order can have multiple records ,each record contains a product, we consider these records form one single data point which is one order. we consider different customer_id as the `user`. I think we currently implement the logic that selecting `prediction_target=vendor_cuisine`  is considering `vendor_id||primary_cuisine` as the `item`. 
  2. We consider `primary_cuisine` as the item's category. Therefore even we have maybe thousands of items, there are limited categorys(limited number of primary cuisine of vendors)
  3. we want to only work on repeated dataset. Now we switch to predict repeated orders. In order to do this, we need to first build a user filter first, we still consider training data contains user's full historical items, testing data, different order present different test cases. We now only consider users:
  a) contains more than 5 historical items, meaning at least 5 orders
  b) only keep test orders whose vendor was visited in the training dataset for the same user. A user can have multiple test orders, but we only keep those test orders whose vendor is one of the vendor within the hisotrical items in its user training data.
  think carefully about design this filtering and keep the filtered dataset somewhere we can cache it, test it throughtly.
  4. We want the first round to be based on the lightGCN trainined based on item categories, that's being said, after filtering out training data, based on rule that there should be at least 5 histoircal items, we construct the lightGCN ground of customer -- item-categories, which is basically customer --- primary cuisine. Primary cuisine is much less than vendor_id|| primary cuisine. 
  5. Now in first round, when we talk to LLM, we present still the user's historical items information, will be in format of `vendor_id || primary_cuisine || (weekday, hour) `, and the top 10 primary cuisines that the user historically purchased, and ranked them by the  calculation result based on lightGCNs training for this user(each item categroy (primary cuisine) and user are both emebedding, we use dot product to calculate their relationship score), and we will give the test order's weekday, hour so that llm should predict the top chioces of 3 primary cuisine
  6. base on the first round's 3 primary cuisine, we need to select a list of candidate vendors. One good way is that, we have to pre-compute and cache these informations:
  a) a vendor's metadata: geohash, primary cuisine.
  b) geohash -> list of cuisine -> list of vendors. meaning, with in the trianing dataset, we group vendors first by geohash, then by its primary cuisine. then we can easily use this map to find out that given a geohash, we know what vendors are under this geohash, and if we give a list of cuisines, we know what restaurant are there.

  once we have these information pre-computed, for preparing candidate items (vendors), we will go into the user's historical items, given the test order's geohash information, we only select the vendors that meet these conditions:
  a) same geohash as the test order, meaning we limit the location of the vendors
  b) vendors whose primary cuisine is in the list of 3 primary cuisines we selected from round1 response. 

  7. after we select the candidate vendors, we wnat to present collobrative information for round 2 prediction. What we do is, we use swing similarity between users, find the top 5 similar users for this test order user, and then within their historical vendors (from training dataset), we select the records (vendor, primary cuisine, weekyday+hour) whose primary cuisine are within the round 1 result top 3 primary cuisine. adding these records as the collborative filtering information in the round 2 request. at most 20 candidate vendors

  8. therefore, round 2 requests contains:
  a) round 1 predicition of primary cuisine, with order
  b) selected vendors candidates given by step 6
  c) additionaly information of simialr user from step 7, we should controle the similiar users' selected historical records no more than 5 records per similar user.
  d) provide the order weekday+hour, and ask LLM to rank the candidate vendors.


 ## requirement
 1. make sure we keep each of the number criterial, like x records per user, top x simialr user, being configable in the config file.
 2. after we make the code implementation of each update. we should plan to test it before we move the coding in next task
 3. we can see there are multiple things we can pre-compute and cache, make the caching as an option in config. and we can do those pre-compute like filtering records in training, in testing data, pre-compute the swing, lightGCN, geohash -> primary cuisine -> vendors mapping, and any others. exlicity in config marking them as using cache true, then we will compute and cache them. if use cache as false, we recompute the cache
 4. we still use the same way of calculating the metrics. we care about the hit rate and ncds, and also the #1 #3 #5 metrics
 5. still using the stage like design we have, in final stage, we still keep the detailed json that contains everything like request to LLM, response from LLM. 

### result:
doc/design/Repeated_Dataset_Evaluation_Plan260131.md



 ## later improvement
 1. write comment in the place explaining why we use vendor geohash to filter vendors before round2. because we want to preserve the real app secanrio that we are recommending vendors based on area
 2. in the details result outputs/202601310519/stage9_repeat_detailed.json, I want you to also add the historical items tuple (vendor id, cuisine, time), associated recorsd we pass into round2 for each similar user, also in tuple format. also record the time spend in waiting llm response.
 3. after we fix this, copy the config file using cp command, then just change the model to `google/gemini-2.0-flash-001` and update n_samples as 500. then using multi-thread, multi worker like 50. run the code



   ┌──────────┬────────────────┬──────────────────┬──────────────────┬───────────┐                  
  │  Metric  │ Gemini 3 Flash │ Gemini 2.5 Flash │ Gemini 2.0 Flash │ Kimi K2.5 │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Samples  │ 200            │ 200              │ 500              │ 199       │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Hit@1    │ 0.5250         │ 0.5000           │ 0.4940           │ 0.2613    │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Hit@3    │ 0.6600         │ 0.6400           │ 0.6060           │ 0.3769    │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Hit@5    │ 0.6750         │ 0.6400           │ 0.6440           │ 0.4523    │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ MRR      │ 0.5906         │ 0.5673           │ 0.5620           │ 0.3440    │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ GT found │ 68.0%          │ 66.0%            │ 72.4%            │ 60.8%     │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Avg rank │ 1.4            │ 1.5              │ 2.5              │ 4.24      │                  
  ├──────────┼────────────────┼──────────────────┼──────────────────┼───────────┤                  
  │ Avg time │ 5.5s           │ 8.8s             │ 3.1s             │ ~5s       │
  └──────────┴────────────────┴──────────────────┴──────────────────┴───────────┘

## Jan 31

### Dataset-Prefixed Outputs

Refactored output structure to support multiple datasets without collision:
- All outputs now go to `outputs/{dataset_name}/` (e.g., `outputs/data_se/`, `outputs/data_sg/`)
- Auto-detect data file names in `enriched_loader.py` (no more hardcoded `orders_sg_train.txt`)
- Fixed hardcoded file names in `workflow_runner.py` stage 1 cache + log messages
- Updated all configs (`workflow_config_qwen32_linux.yaml`, `workflow_config_gemini_repeat.yaml`, etc.)
- Created new config: `workflow_config_qwen32_linux_sg.yaml` for data_sg dataset
- Moved existing root-level output files to `outputs/data_se/`

### data_sg Initial Test (5 samples, qwen3-32b, repeat eval)

Dataset stats: 3.4M training rows, 591K test rows, 78 cuisines, 105K users, 7153 vendors

| Metric   | data_sg (5 samples) |
|----------|---------------------|
| Hit@1    | 0.6000              |
| Hit@3    | 0.6000              |
| Hit@5    | 0.8000              |
| NDCG@5   | 0.6861              |
| MRR      | 0.6500              |
| GT found | 80.0%               |
| Avg rank | 1.8                 |
| Errors   | 0/5                 |



## improve metrics
 in stage 9'/home/zhenkai/personal/Projects/AgenticRecommender/outputs/data_sg/202601310805/stage9  
  _repeat_detailed.json' can we update the recording code that update that 1) what is the ground     
  truth rank in round1. 2) what is the ground truth rank in the provide lightGCN. 3) can we          
  document in '/home/zhenkai/personal/Projects/AgenticRecommender/outputs/data_sg/202601310805/stag  
  e9_repeat_results.json' that round 1's metrics, round2's metrics. seeing the improve of the        
  metrics from round1 to round2. we want the same set of metrics currently defined.  