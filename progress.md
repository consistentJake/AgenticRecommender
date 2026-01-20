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
