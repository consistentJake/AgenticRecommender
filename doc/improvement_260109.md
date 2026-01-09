## issues
1. in /home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/agents/analyst.py, it is trying to analyze the user. Instead, we should add the prediction funcation.
2. The prediction ability I want to add is already written in /home/zhenkai/personal/Projects/AgenticRecommender/finetune/scripts/utils.py, to_chat_messages function can read example and then build the prompt.
3. there are datasets. one is movielens, example is here 
```
{"instruction": "Predict whether the user will like the candidate movie. Answer only with Yes or No.", "input": "User's last 15 watched movies:\n1. Quick Change (1990) (rating ≈ 4.0)\n2. Ferris Bueller's Day Off (1986) (rating ≈ 4.5)\n3. Weekend at Bernie's (1989) (rating ≈ 1.5)\n4. Léon: The Professional (a.k.a. The Professional) (Léon) (1994) (rating ≈ 3.5)\n5. House of Flying Daggers (Shi mian mai fu) (2004) (rating ≈ 3.5)\n6. Pianist, The (2002) (rating ≈ 5.0)\n7. Anchorman: The Legend of Ron Burgundy (2004) (rating ≈ 2.5)\n8. Ocean's Twelve (2004) (rating ≈ 3.0)\n9. Shaun of the Dead (2004) (rating ≈ 4.5)\n10. Manchurian Candidate, The (2004) (rating ≈ 3.0)\n11. 21 Grams (2003) (rating ≈ 3.0)\n12. Star Wars: Episode II - Attack of the Clones (2002) (rating ≈ 1.5)\n13. Star Wars: Episode III - Revenge of the Sith (2005) (rating ≈ 2.5)\n14. Star Wars: Episode V - The Empire Strikes Back (1980) (rating ≈ 3.5)\n15. Punch-Drunk Love (2002) (rating ≈ 4.5)\n\nCandidate movie:\nYour Friends and Neighbors (1998)\n\nShould we recommend this movie to the user? Answer Yes or No.", "output": "No", "system": "You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.", "history": []}

```
training file is here `/home/zhenkai/personal/Projects/AgenticRecommender/finetune/data/movielens_qwen3/train.jsonl`
4. you would need to make the analyst.py has the ability to do prediction, given a training example. You can read this code to understand how we prepare the dataset /home/zhenkai/personal/Projects/AgenticRecommender/finetune/scripts/prepare_movielens.py

5. therefore I want to upgrade the code in folder /home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/agents, to actually do prediction on the next item. don't use _analyze_user function in /home/zhenkai/personal/Projects/AgenticRecommender/agentic_recommender/agents/analyst.py anymore
6. make sure the agent workflow code is running able eventually by passing a example. make a reform plan and execute 


