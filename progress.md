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
```


## Jan 13 improvement plan
the workflow stars from do `agentic_recommender/workflow/workflow_runner.py --config agentic_recommender/workflow/workflow_config_linux.yaml`. you will find the logic and code starting from here.
Followings are several points I want to make updates, please complete them one by one but think about how to update the code in a systematic way.
1. in preparing the enriched_user data, I want to make change of how we record the history purchase behavior, we should not just record the list of cuisine, we should also for each cuisine, we record the vendor id, the hour, the week day the purchase happen. 
2. therefore during prediction task, we want to predict the next item after the user's history, the hour and the week day should be given, and then we predict the cuisine. 
3. when we are storing the processed data, I like the way we create preview like outputs/stage1_merged_preview.json, so that i can talk a look of the data generated rather than we have so many big file and hard to look at. so keep the full data json file is okay, but also add a step to create a small preview file for each json file we generated.



