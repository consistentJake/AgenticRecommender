## set up folder
git clone https://github.com/wzf2000/MACRec.git
git clone https://github.com/dh-r/LLM-Sequential-Recommendation.git
git clone https://github.com/Yu-Qi-hang/ThinkRec



So I would like you to write out the design document. So let’s what do you want to do? What I want you to These are what very clearly was Thank Call the working phones. So we want to have 2 kind of datasets. First is the delivery hero, and second one is Hero Home Beauty. Which is the same of the LLM-sequential-recommendations, (paper  is papers/Improving Sequential Recommendations with LLMs.pdf, code is in previousWorks/LLM-Sequential-Recommendation, previous analysis is written in LLM_Sequential_Recommendation_Analysis.md). Paper. And with that, the true data set, We Follow the Same way To prepare the datasets. And I want it to focus on one Recommendation task is the sequential recommendations. In the next We want to make it as a agentic So this agent to workflow We can follow the MacRec  Paper (paper is papers/MacRec.pdf, code is in previousWorks/MACRec folder, previous analysis is written in MACRec_Analysis.md) The MacRec paper will have the manager and Analyst. Understanding how MacRec is trying to strcutre its agents, and how their communication work. Then its recommendation result will be Verified, metrics  you can also find it in the LLM-sequential-recommendations. Using the same  matrix They use. And later what we want to do is we add a reflector section So reflector is following the idea in the MarRec. And What will, more advanced feature we want to do is for the for the for the training, but for the for the the core of the model. The idea is from ThinkRec (paper is papers/thinkRec.pdf, code is previousWorks/ThinkRec, previous analysis is written in ThinkRec_Technical_Analysis.md). I want it to be trained with both recommendation task and the reasoning task, learn its design and loss function. Try to break down the tasks, and add the right reference of local paper and code, think hard in each task, of complete the design document step by step. 


## dataset download
from previousWorks/LLM-Sequential-Recommendation/beauty/README.md

The dataset was found [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  in "per-category files". The necessary files are the Beauty [reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz) and the Beauty [metadata](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz). Note this is the older version of the Amazon Beauty dataset. This is intentional, because this way it matches the dataset with most research papers (BERT4Rec, SASRec) using Beauty.
