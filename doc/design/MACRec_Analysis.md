# MACRec: Multi-Agent Collaboration Framework for Recommendation - Technical Analysis

## Overview

MACRec (Multi-Agent Collaboration Framework for Recommendation) is a comprehensive framework that leverages Large Language Models (LLMs) to build recommendation systems through multi-agent collaboration. Unlike previous work that primarily focuses on user/item simulation with agents, MACRec directly tackles recommendation tasks through the collaborative efforts of various specialized agents.

## Architecture Overview

### Core Agent Types and Their Roles

#### 1. Manager Agent (`/macrec/agents/manager.py`)
The Manager acts as the central orchestrator of the entire system, implementing a two-stage decision-making process:

**Key Features:**
- **Two-stage LLM architecture**: Separate LLMs for "thought" and "action" phases
- **Token limit management**: Prevents context overflow by monitoring token counts
- **Collaborative workflow coordination**: Manages interactions with all other agents

### Two-Stage LLM Architecture - Detailed Analysis

**What this helps in the system:**
- **Specialized reasoning**: The "thought" LLM focuses purely on reasoning and analysis
- **Structured action generation**: The "action" LLM generates properly formatted actions
- **Better token management**: Each LLM can be optimized for its specific task
- **Improved consistency**: Actions follow stricter formatting rules while thoughts can be more free-form

**Implementation (`/macrec/agents/manager.py:14-33, 76-85`):**
```python
class Manager(Agent):
    def __init__(self, thought_config_path: str, action_config_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Two separate LLMs with potentially different configurations
        self.thought_llm = self.get_LLM(thought_config_path)
        self.action_llm = self.get_LLM(action_config_path)
        self.json_mode = self.action_llm.json_mode
        
    def _prompt_thought(self, **kwargs) -> str:
        thought_prompt = self._build_manager_prompt(**kwargs)
        thought_response = self.thought_llm(thought_prompt)  # Uses thought LLM
        return format_step(thought_response)
        
    def _prompt_action(self, **kwargs) -> str:
        action_prompt = self._build_manager_prompt(**kwargs)
        action_response = self.action_llm(action_prompt)    # Uses action LLM
        return format_step(action_response)
        
    def forward(self, stage: str, *args, **kwargs) -> str:
        if stage == 'thought':
            return self._prompt_thought(**kwargs)
        elif stage == 'action':
            return self._prompt_action(**kwargs)
```

**Workflow Integration (`/macrec/systems/collaboration.py:97-115`):**
```python
def think(self):
    # Uses thought LLM for reasoning
    thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
    self.scratchpad += ' ' + thought
    
def act(self) -> tuple[str, Any]:
    # Uses action LLM for structured action generation
    action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
    action_type, argument = parse_action(action, json_mode=self.manager.json_mode)
    return action_type, argument
```

**Prompt Template (`/config/prompts/manager_prompt/reflect_analyse_search_interpret.json:4`):**
```
"Solve a {task_type} task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation. 
Action can be 2 types:
(1) Analyse[user/item, id] - ask Analyst to analyze preferences/features
(2) Search[requirements] - ask Searcher to find information  
(3) Interpret[content] - ask Task Interpreter to summarize content
(4) Finish[response] - finish task and return response"
```

**Location**: `/macrec/agents/manager.py:10-94`

#### 2. Analyst Agent (`/macrec/agents/analyst.py`)
Specializes in analyzing user and item characteristics through access to information databases and interaction history.

**Tools Available:**
- **InfoDatabase**: Retrieves user profiles and item attributes
- **InteractionRetriever**: Accesses historical user-item interactions

**Available Commands:**
```python
# Commands the Analyst can execute
"UserInfo[id]"          # Get user profile information
"ItemInfo[id]"          # Get item attributes  
"UserHistory[id, k]"    # Get user's k most recent interactions
"ItemHistory[id, k]"    # Get item's k most recent user interactions
"Finish[result]"        # Return analysis result
```

**Location**: `/macrec/agents/analyst.py:8-198`

#### 3. Searcher Agent (`/macrec/agents/searcher.py`)
Handles external information retrieval using search tools like Wikipedia.

**Core Functions:**
```python
# Available search commands
"Search[query]"         # Search for relevant information
"Lookup[title, term]"   # Look up specific terms in documents
"Finish[result]"        # Return search results
```

**Location**: `/macrec/agents/searcher.py:9-119`

#### 4. Reflector Agent (`/macrec/agents/reflector.py`)
Implements multiple reflection strategies to improve answer quality through iterative refinement.

### Reflector Agent - Detailed Analysis

**Different Reflection Strategies (`/macrec/agents/reflector.py:11-18`):**
```python
class ReflectionStrategy(Enum):
    NONE = 'base'                                    # No reflection
    LAST_ATTEMPT = 'last_trial'                     # Store previous attempt
    REFLEXION = 'reflection'                        # LLM-based analysis  
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflection'  # Combined
```

**Implementation of Each Strategy (`/macrec/agents/reflector.py:87-105`):**
```python
def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
    if self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT:
        # Simply stores the last attempt as context
        self.reflections = [scratchpad]
        self.reflections_str = format_last_attempt(input, scratchpad, 
                                                 self.prompts['last_trial_header'])
        
    elif self.reflection_strategy == ReflectionStrategy.REFLEXION:
        # Uses LLM to analyze and improve
        self.reflections.append(self._prompt_reflection(input=input, scratchpad=scratchpad))
        self.reflections_str = format_reflections(self.reflections, 
                                                header=self.prompts['reflection_header'])
        
    elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION:
        # Combines both approaches
        self.reflections_str = format_last_attempt(input, scratchpad, 
                                                 self.prompts['last_trial_header'])
        self.reflections = self._prompt_reflection(input=input, scratchpad=scratchpad)
        self.reflections_str += format_reflections(self.reflections, 
                                                  header=self.prompts['reflection_last_trial_header'])
```

**Example Reflection Prompts (`/config/prompts/agent_prompt/reflector.json`):**

**Text Mode Prompt:**
```
"You are an advanced reasoning agent that can improve based on self reflection. 
You will be given a previous reasoning trial in which you were given a task to complete. 
Firstly, you should determine if the given answer is correct. 
Then, provide reasons for your judgement. 
Possible reasons for failure may be guessing the wrong answer using Finish[<answer>] 
or using a wrong format for action. 
In a few sentences, discover the potential problems in your previous reasoning trial 
and devise a new, concise, high level plan that aims to mitigate the same failure."
```

**JSON Mode Examples:**
```json
{"correctness": false, "reason": "The potential problem ..."}
{"correctness": true, "reason": "There is no problem with the agent's plan ..."}
```

**Reflection Headers:**
- `reflection_header`: "You have attempted to complete a following task before. The following reflection(s) give a new plan..."
- `last_trial_header`: "You have attempted to complete the following task before. Below is the last trial..."

**Location**: `/macrec/agents/reflector.py:20-106`

#### 5. Task Interpreter Agent (`/macrec/agents/interpreter.py`)
Translates conversational dialogs into structured recommendation tasks, especially crucial for conversational recommendation scenarios.

### Task Interpreter Agent - Detailed Analysis

**What the Interpreter Does:**
The Interpreter agent converts conversational or unclear inputs into structured, actionable prompts for the recommendation system.

**Available Commands (`/macrec/agents/interpreter.py:76-92`):**
```python
"Summarize[]"     # Get summary of long/complex requirements
"Finish[prompt]"  # Return structured task prompt
```

**Processing Flow:**
```python
def forward(self, input: str, *args, **kwargs) -> str:
    tokens = input.split()
    if len(tokens) > 100:
        truncated_input = '...' + ' '.join(tokens[-100:])  # Keep last 100 words
    else:
        truncated_input = input
    
    while not self.is_finished():
        command = self._prompt_interpreter(input=truncated_input)
        self.command(command, input=input)
```

**Input/Output Example:**

**Input** (conversational):
```
"I'm very interested in watching movie. But recently I couldn't find a movie that satisfied me very much. Do you know how to solve this?"
```

**Expected Output** (structured prompt):
```
"Please recommend movies for a user who enjoys watching movies but hasn't found satisfying options recently. Provide movie recommendations based on their preferences."
```

**Prompt Template (`/config/prompts/agent_prompt/interpreter.json:4`):**
```
"I want you to act as an prompt interpreter. I will give you the basic requirements, 
and you should give a concise and clear prompt for the system to generate the appropriate answer.
You can use 2 type of commands to do this:
(1) Summarize[], which will give you a summary of requirements by some text summarization tools.
(2) Finish[prompt], which returns the prompt you generated and finishes the task."
```

**Command Examples:**
```python
# Text format
"Summarize[]"
"Finish[Please tell me the weather in New York City.]"

# JSON format  
{"type": "Summarize", "content": ""}
{"type": "Finish", "content": "Please tell me the weather in New York City."}
```

## Agentic Flow Implementation

### System Orchestration (`/macrec/systems/collaboration.py`)

The `CollaborationSystem` class orchestrates the entire multi-agent workflow:

**Core Workflow Steps:**
```python
def step(self):
    self.think()                    # Manager reasoning phase
    action_type, argument = self.act()  # Manager action decision
    self.execute(action_type, argument) # Execute action with appropriate agent
    self.step_n += 1

def execute(self, action_type: str, argument: Any):
    if action_type.lower() == 'analyse':
        observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
    elif action_type.lower() == 'search':
        observation = self.searcher.invoke(argument=argument, json_mode=self.manager.json_mode)
    elif action_type.lower() == 'interpret':
        observation = self.interpreter.invoke(argument=argument, json_mode=self.manager.json_mode)
    elif action_type.lower() == 'finish':
        observation = self.finish(parse_result['answer'])
```

**Location**: `/macrec/systems/collaboration.py:157-217`

### Agent Communication Mechanism

#### 1. **Invocation Pattern**
Agents communicate through a standardized `invoke()` method:

```python
# From collaboration.py:132-134
self.log(f':violet[Calling] :red[Analyst] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager)
observation = self.analyst.invoke(argument=argument, json_mode=self.manager.json_mode)
log_head = f':violet[Response from] :red[Analyst] :violet[with] :blue[{argument}]:violet[:]\n- '
```

#### 2. **Structured Action Parsing**
Actions are parsed from Manager outputs using standardized formats:

```python
# JSON format example
{"type": "Analyse", "content": ["user", 524]}
{"type": "Search", "content": "movie recommendations"}
{"type": "Finish", "content": "recommendation result"}

# Text format example  
"Analyse[user, 524]"
"Search[movie recommendations]"
"Finish[recommendation result]"
```

#### 3. **Shared Context Management**
All agents share access to:
- Current data sample (user_id, item_id, etc.)
- Task type and configuration
- System-wide logging and web demo functionality

## System-Wide Logging Mechanism

### Logging Implementation

**Location**: `/macrec/systems/base.py:93-114`

```python
def log(self, message: str, agent: Optional[Agent] = None, logging: bool = True) -> None:
    if logging:
        logger.debug(message)  # Standard logging
        
    if self.web_demo:
        if agent is None:
            role = 'Assistant'
        else:
            role = agent.__class__.__name__  # Get agent type (Manager, Analyst, etc.)
            
        # Format message with agent identity and color coding
        final_message = f'{get_avatar(role)}:{get_color(role)}[**{role}**]: {message}'
        
        # Special formatting for non-manager agents (indentation)
        if 'manager' not in role.lower() and 'assistant' not in role.lower():
            messages = final_message.split('\n')
            messages = [f'- {messages[0]}'] + [f'  {message}' for message in messages[1:]]
            final_message = '\n'.join(messages)
            
        self.web_log.append(final_message)
        st.markdown(f'{final_message}')  # Display in web interface
```

### How Agents Use Logging

**1. Manager Agent** - Logs thoughts and actions:
```python
# From collaboration.py:103, 155
self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager)
self.log(f'{log_head}{observation}', agent=self.manager, logging=False)
```

**2. Analyst Agent** - Logs tool usage:
```python
# From analyst.py:84, 91, 115
log_head = f':violet[Look up UserInfo of user] :red[{query_user_id}]:violet[...]\n- '
log_head = f':violet[Look up ItemInfo of item] :red[{query_item_id}]:violet[...]\n- '
log_head = f':violet[Look up UserHistory of user] :red[{query_user_id}] :violet[with at most] :red[{k}] :violet[items...]\n- '
```

**3. Reflector Agent** - Logs reflections:
```python
# From reflector.py:81-83
if self.json_mode:
    self.system.log(f"[:violet[Reflection]]:\n- `{self.reflection_output}`", agent=self, logging=False)
else:
    self.system.log(f"[:violet[Reflection]]:\n- {self.reflection_output}", agent=self, logging=False)
```

**4. Inter-agent Communication Logging:**
```python
# From collaboration.py:132, 134, 139, 141
self.log(f':violet[Calling] :red[Analyst] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager)
log_head = f':violet[Response from] :red[Analyst] :violet[with] :blue[{argument}]:violet[:]\n- '
self.log(f':violet[Calling] :red[Searcher] :violet[with] :blue[{argument}]:violet[...]', agent=self.manager)
log_head = f':violet[Response from] :red[Searcher] :violet[with] :blue[{argument}]:violet[:]\n- '
```

**Purposes of Logging:**
- **Debugging and monitoring**: Track agent behavior and decision-making processes
- **Web demo visualization**: Provide real-time feedback in the user interface
- **Agent identification**: Color-coded messages to distinguish between different agents
- **Communication tracking**: Log inter-agent communications and responses
- **Performance analysis**: Monitor workflow efficiency and bottlenecks

## Dataset Preparation and Examples

### Supported Datasets

#### 1. MovieLens 100K (`/macrec/dataset/ml100k.py`)

**Data Processing Pipeline:**
```python
def process_data(dir: str, n_neg_items: int = 9):
    # Download raw data from GroupLens
    download_data(dir)
    
    # Read and process components
    data_df, item_df, user_df, genre_df = read_data(os.path.join(dir, "raw_data"))
    user_df = process_user_data(user_df)      # Process user profiles
    item_df = process_item_data(item_df)      # Process item attributes
    
    # Generate train/dev/test splits with negative sampling
    train_df, dev_df, test_df = process_interaction_data(data_df, n_neg_items)
    
    # Append historical interaction information
    dfs = append_his_info([train_df, dev_df, test_df], neg=True)
```

**User Profile Template:**
```python
# From ml100k.py:48-52
template = PromptTemplate(
    template='Age: {age}\nGender: {gender}\nOccupation: {occupation}',
    input_variables=['age', 'gender', 'occupation']
)
user_df['user_profile'] = user_df[input_variables].apply(lambda x: template.format(**x), axis=1)
```

**Item Attributes Template:**
```python
# From ml100k.py:79-83
template = PromptTemplate(
    template='Title: {title}, Genres: {genre}',
    input_variables=['title', 'genre']
)
item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
```

#### 2. Amazon Product Data (`/macrec/dataset/amazon.py`)

**Data Processing Features:**
- Downloads data from Stanford SNAP datasets
- Processes product metadata (brand, price, categories)
- Handles user reviews and ratings
- Creates rich item attribute descriptions

**Item Attribute Template:**
```python
# From amazon.py:84-88
template = PromptTemplate(
    template='Brand: {brand}, Price: {price}, Categories: {categories}',
    input_variables=['brand', 'price', 'categories']
)
item_df['item_attributes'] = item_df[input_variables].apply(lambda x: template.format(**x), axis=1)
```

### Data Format Example

After processing, each interaction sample contains:
```python
{
    'user_id': 524,
    'item_id': 181,
    'rating': 4.0,
    'user_profile': 'Age: 25\nGender: male\nOccupation: technician',
    'target_item_attributes': 'Title: Toy Story, Genres: Animation|Children|Comedy',
    'history': 'Title: Star Wars, Genres: Action|Adventure|Sci-Fi (rating: 5)\nTitle: ...',
    'candidate_item_id': [181, 95, 234, 567, ...],  # target + negatives
    'candidate_item_attributes': '181: Title: Toy Story, Genres: Animation|Children|Comedy\n95: ...'
}
```

## Recommendation Tasks and Data Preparation

### 1. Rating Prediction (RP)

**Agent Configuration:**
- **Required**: Manager, User Analyst, Item Analyst
- **Optional**: Reflector

**Data Prompt Template:**
```python
# From generation.py:38-44
data_prompt.format(
    user_id=df['user_id'][i],
    user_profile=df['user_profile'][i], 
    history=df['history'][i],
    target_item_id=df['item_id'][i],
    target_item_attributes=df['target_item_attributes'][i]
)
```

**Workflow:**
1. Manager analyzes the prediction task
2. Calls User Analyst to understand user preferences
3. Calls Item Analyst to understand item characteristics  
4. Predicts rating based on user-item compatibility

### 2. Sequential Recommendation (SR)

**Agent Configuration:**
- **Required**: Manager, User Analyst
- **Optional**: Reflector
- **Special**: Item Analyst less critical due to large candidate sets

**Data Prompt Template:**
```python
# From generation.py:48-53
data_prompt.format(
    user_id=df['user_id'][i],
    user_profile=df['user_profile'][i],
    history=df['history'][i], 
    candidate_item_attributes=df['candidate_item_attributes'][i]
)
```

**Workflow:**
1. Manager analyzes user's sequential behavior
2. User Analyst identifies long-term and short-term preferences
3. Reflector helps avoid formatting errors in ranking output
4. Returns ranked list of candidate items

## Sequential Recommendation - Candidate Pool Generation

### Candidate Pool Creation

The system **does NOT generate random items**. Instead, it uses a structured approach:

**Location**: `/macrec/dataset/ml100k.py:162, 168` and `/macrec/dataset/amazon.py:187, 193`

```python
# 1. Create candidate pool: target item + negative samples
df['candidate_item_id'] = df.apply(lambda x: [x['item_id']] + x['neg_item_id'], axis=1)

# 2. Shuffle the order to avoid positional bias
def shuffle_list(x):
    random.shuffle(x)
    return x

df['candidate_item_id'] = df['candidate_item_id'].apply(lambda x: shuffle_list(x))
```

### Negative Sampling Strategy

**Location**: `/macrec/dataset/ml100k.py:108-117`

```python
def negative_sample(df):
    neg_items = np.random.randint(1, n_items + 1, (len(df), n_neg_items))  # Default n_neg_items=9
    for i, uid in enumerate(df['user_id'].values):
        user_clicked = clicked_item_set[uid]  # Items user has already interacted with
        for j in range(len(neg_items[i])):
            # Ensure negatives are: 1) Not clicked by user, 2) Not duplicates
            while neg_items[i][j] in user_clicked or neg_items[i][j] in neg_items[i][:j]:
                neg_items[i][j] = np.random.randint(1, n_items + 1)
        assert len(set(neg_items[i])) == len(neg_items[i])  # Verify no duplicates
    df['neg_item_id'] = neg_items.tolist()
    return df
```

### Final Candidate Format

**Location**: `/macrec/dataset/ml100k.py:172-178`

```python
def candidate_attr(x):
    candidate_item_attributes = []
    for item_id, item_attributes in zip(x, item_df.loc[x]['item_attributes']):
        candidate_item_attributes.append(f'{item_id}: {item_attributes}')
    return candidate_item_attributes

df['candidate_item_attributes'] = df['candidate_item_id'].apply(lambda x: candidate_attr(x))
df['candidate_item_attributes'] = df['candidate_item_attributes'].apply(lambda x: '\n'.join(x))
```

### Data Prompt for Sequential Recommendation

**Location**: `/config/prompts/data_prompt/sr.json:4`

```
"[user_id]: user_{user_id}
[user_profile]:
{user_profile}
[historical interactions]:
{history}
According to user_{user_id}'s preference, please give a rank order of the following candidates (with the format [id]: attribute): 
{candidate_item_attributes}"
```

### Example Candidate Pool

```
181: Title: Toy Story, Genres: Animation|Children|Comedy
95: Title: Broken Arrow, Genres: Action|Thriller
234: Title: Nell, Genres: Drama
567: Title: Batman Forever, Genres: Action|Adventure|Crime
889: Title: Pocahontas, Genres: Animation|Children|Musical
...
```

**Expected Output Format:**
```json
[181, 567, 95, 889, 234, ...]  // Ranked list of item IDs
```

The system generates ranked recommendations by having the Manager agent (with help from User Analyst) analyze the user's preferences against this pre-defined candidate pool, rather than generating items from scratch.

### 3. Explanation Generation (EG)

**Agent Configuration:**
- **Required**: Manager, User Analyst, Item Analyst, Searcher
- **Optional**: Reflector

**Enhanced Information Gathering:**
- Searcher retrieves external knowledge (e.g., director information)
- Enables richer, more informative explanations

### 4. Conversational Recommendation (CR)

**Agent Configuration:**
- **Required**: Manager, Task Interpreter, Searcher
- **Optional**: User Analyst, Item Analyst, Reflector

**Special Features:**
```python
# From collaboration.py:182-188
def interprete(self) -> None:
    if self.task == 'chat':
        assert self.interpreter is not None, 'Interpreter is required for chat task.'
        self.manager_kwargs['task_prompt'] = self.interpreter(input=self.chat_history)
```

**Workflow:**
1. Task Interpreter converts dialog to structured task
2. Searcher handles unknown information requests
3. Manager provides conversational responses
4. Maintains dialog history for context

## Prompt Management System

### Hierarchical Prompt Structure

**Configuration Levels:**
```python
# From base.py:81-84
self.prompts = read_prompts(self.config['agent_prompt'])          # Agent-specific prompts
self.prompts.update(read_prompts(self.config['data_prompt']))     # Data formatting prompts  
self.prompts.update(read_prompts(self.config['task_agent_prompt'])) # Task-specific prompts
```

**Prompt Locations:**
- **Agent Prompts**: `/config/prompts/agent_prompt/` - Core agent behavior
- **Manager Prompts**: `/config/prompts/manager_prompt/` - Manager coordination
- **Data Prompts**: `/config/prompts/data_prompt/` - Task data formatting

### Example Manager Prompt Configuration

**File**: `/config/prompts/manager_prompt/analyse.json`
```json
{
    "manager_prompt": {
        "type": "template",
        "content": "Solve a {task_type} task with interleaving Thought, Action, Observation steps..."
    },
    "valid_action_example": {
        "type": "raw", 
        "content": "Finish[{finish}]\nAnalyse[user, 524]\nAnalyse[item, 955]"
    }
}
```

### Example Analyst Prompt Configuration

**File**: `/config/prompts/agent_prompt/analyst.json`
```json
{
    "analyst_prompt": {
        "type": "template",
        "content": "I want you to act as an analyst and help me analyze the given {analyse_type} {id}..."
    },
    "analyst_examples": {
        "type": "raw",
        "content": "UserInfo[123]\nItemInfo[456]\nUserHistory[123, 3]\nFinish[analysis result]"
    }
}
```

## Key Implementation Insights

### 1. Modular Agent Design
- Each agent inherits from base `Agent` or `ToolAgent` classes
- Standardized interfaces for LLM integration and tool access
- Consistent command parsing and execution patterns

### 2. Flexible System Configuration
- JSON-based configuration for easy customization
- Agent selection based on task requirements
- Support for different LLM backends (OpenAI, open-source)

### 3. Robust Communication Protocol
- Structured action formats (JSON and text)
- Standardized invoke/observe patterns
- Comprehensive logging and web demo support

### 4. Scalable Data Processing
- Efficient negative sampling for recommendation tasks
- Historical context management
- Template-based attribute formatting

### 5. Multi-Modal Task Support
- Rating prediction, sequential recommendation, explanation generation
- Conversational recommendation with dialog management
- Extensible framework for new recommendation scenarios

## Fine-Tuning and RLHF Training Implementation

### Overview of MACRec's Fine-Tuning Approach

MACRec implements a comprehensive **Reinforcement Learning from Human Feedback (RLHF)** training pipeline using **Proximal Policy Optimization (PPO)** to fine-tune LLMs for recommendation tasks. This approach goes beyond traditional supervised learning by incorporating reward signals that align model outputs with recommendation objectives.

**Important Note**: The fine-tuning specifically targets the **Manager Agent's LLMs** (both thought and action models), NOT the entire multi-agent system. Other agents (Analyst, Searcher, Reflector, Interpreter) continue using their base pre-trained LLMs without fine-tuning.

### Fine-Tuning Architecture

#### Fine-Tuning Target: Manager Agent's LLMs

**What Gets Fine-Tuned**:
- **Manager's Thought LLM**: Used for reasoning and situation analysis (`/macrec/agents/manager.py:76-79`)
- **Manager's Action LLM**: Used for generating structured actions (`/macrec/agents/manager.py:81-85`)

**What Does NOT Get Fine-Tuned**:
- Other agents (Analyst, Searcher, Reflector, Interpreter) - they use base pre-trained models
- Framework coordination logic - remains unchanged
- Agent communication protocols - stay the same

#### 1. Training Pipeline (`/macrec/tasks/rlhf.py`)

**Core Training Class**:
```python
class RLHFTrainingTask(Task):
    def train(self, epochs: int = 1):
        base_dir = os.path.join('ckpts/', str(int(time.time())))
        os.makedirs(base_dir, exist_ok=True)
        for epoch in range(epochs):
            for batch_id, batch in tqdm(enumerate(self.trainer.dataloader)):
                query_tensors, response_tensors = batch['input_ids'], batch['output_ids']
                rewards = batch['rewards']
                # Core PPO training step - fine-tunes the Manager's LLM
                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                self.trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])
            self.trainer.save_pretrained(os.path.join(base_dir, f'epoch-{epoch}'))
```

**Training Configuration (`/config/training/ppo-main.json`)**:
```json
{
    "model_path": "lmsys/vicuna-7b-v1.5-16k",  // Manager's base LLM to be fine-tuned
    "epochs": 4,
    "ppo_kwargs": {
        "learning_rate": 1.41e-5,
        "mini_batch_size": 2,
        "batch_size": 2,
        "target_kl": 6.0,
        "kl_penalty": "kl"
    },
    "peft_kwargs": {
        "r": 16,                    # LoRA rank for Manager's LLM adaptation
        "lora_alpha": 16,           # LoRA scaling parameter
        "bias": "none",
        "task_type": "CAUSAL_LM"    # Fine-tuning the Manager as a causal LM
    }
}
```

#### 2. Dataset Preparation for RLHF (`/macrec/rl/offline_ppo_dataset.py`)

**OfflinePPODataset Structure**:
```python
class OfflinePPODataset(Dataset):
    def __init__(self, prompts: list[str], responses: list[str], rewards: list[int | float], tokenizer):
        self.prompts = prompts      # Input prompts (user/item context)
        self.responses = responses  # Model responses (recommendations)
        self.rewards = rewards      # Reward signals (feedback quality)
        
    def __getitem__(self, index):
        return {
            'input_ids': self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0),
            'output_ids': self.tokenizer.encode(response, return_tensors='pt').squeeze(0),
            'rewards': torch.tensor(reward, dtype=torch.float16)
        }
```

**Data Format Example**:
```jsonl
{
    "input": "[user_id]: user_524\n[user_profile]: Age: 25\nGender: male...",
    "output": "4.2",
    "reward": 0.75
}
```

### Reward Model Implementation

#### 1. Base Reward Architecture (`/macrec/rl/reward/base.py`)

**Three Types of Reward Functions**:

**A. Delta Reward** (Improvement-based):
```python
class DeltaReward(Reward):
    def reward(self, action1: Any, action2: Any, gt_answer: Any) -> float:
        # Reward = improvement from action1 to action2
        return self.action_reward(action2, gt_answer) - self.action_reward(action1, gt_answer)
```

**B. Reflection Reward** (Correctness-based):
```python
class ReflectionReward(Reward):
    def __init__(self, alpha: float = 16):
        self.alpha = alpha  # Reward magnitude
        
    def reward(self, action1: float, action2: float, gt_answer: float, reflection_output: str) -> float:
        reflections = json.loads(reflection_output)  # {"correctness": bool, "reason": str}
        correctness = self.judge(action1, gt_answer)
        if correctness == reflections['correctness']:
            return self.alpha    # Correct self-assessment
        else:
            return -self.alpha   # Incorrect self-assessment
```

#### 2. Task-Specific Reward Functions

**Rating Prediction Rewards (`/macrec/rl/reward/rp.py`)**:

**V1 - Basic Squared Error**:
```python
class RatingPredictionRewardV1(DeltaReward):
    def action_reward(self, action: float, gt_answer: float) -> float:
        if action < self.lower or action > self.upper:  # Invalid rating
            return self.invalid  # Heavy penalty
        return -(gt_answer - action) ** 2  # Negative squared error
```

**V2 - Enhanced with Validity and Stability**:
```python
class RatingPredictionRewardV2(Reward):
    def __init__(self, invalid: float = -16, alpha: float = 4, gamma: float = 0.25, eta: float = 2):
        # Parameters for handling invalid actions and stability
        
    def reward(self, action1: float, action2: float, gt_answer: float) -> float:
        # Complex reward considering:
        # 1. Validity of both actions
        # 2. Improvement magnitude
        # 3. Stability (unchanged but correct actions)
        original_reward = action2_reward - action1_reward
        return original_reward + np.exp(-np.abs(original_reward) * self.eta) * (self.alpha + action2_reward)
```

**Sequential Recommendation Rewards (`/macrec/rl/reward/sr.py`)**:

**V1 - Position-based Reward**:
```python
class SequentialRecommendationRewardV1(DeltaReward):
    def action_reward(self, action: list[int], gt_answer: int) -> float:
        if gt_answer not in action:
            return 0  # Target item not in ranking
        gt_pos = action.index(gt_answer)
        return 1 / (gt_pos + 1)  # Higher reward for better positions
```

**Reflection Reward**:
```python
class SequentialRecommendationReflectionReward(ReflectionReward):
    def judge(self, action: list[int], gt_answer: int) -> bool:
        return len(action) > 0 and action[0] == gt_answer  # Top-1 accuracy
```

### Training Data Generation Pipeline

#### 1. Feedback Collection (`/macrec/tasks/feedback.py`)

**Two-Step Generation Process**:
```python
class FeedbackTask(GenerationTask, RewardTask):
    @property
    def running_steps(self) -> int:
        return 2  # Generate two responses per sample
        
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict):
        # Store both initial and reflected responses
        record[f"Answer_{step + 1}"] = answer
        if hasattr(self.system, 'reflected') and self.system.reflected:
            record["input"] = self.system.reflector.reflection_input
            record["output"] = self.system.reflector.reflection_output
            
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm):
        # Calculate reward between first and second attempt
        record['reward'] = self.reward_model(
            action1=record["Answer_1"], 
            action2=record["Answer_2"], 
            gt_answer=record["Answer_GT"], 
            reflection_output=record["output"]
        )
```

#### 2. Training Data Structure

**Generated Training Samples**:
```jsonl
{
    "input": "[user_id]: user_524\n[user_profile]: Age: 25\n[historical interactions]: ...",
    "Answer_1": "3.5",                    # Initial response
    "Answer_2": "4.0",                    # Response after reflection
    "Answer_GT": 4.0,                     # Ground truth
    "output": "4.0",                      # Final model output for training
    "reward": 2.75                        # Computed reward signal
}
```

### Training Scripts and Workflow

#### 1. Complete Training Pipeline (`/scripts/train.sh`)

**Step 1: Generate Feedback Data**
```bash
# Generate reflection-based feedback data (500 samples)
python main.py --main Feedback \
    --data_file data/ml-100k/train.csv \
    --system reflection \
    --system_config config/systems/reflection/config_open.json \
    --task rp \
    --feedback_file data/ppo/rp/ml-100k-reflection.jsonl \
    --reward_version reflection
```

**Step 2: Update Rewards**
```bash
# Convert reflection rewards to task-specific rewards
python main.py --main RewardUpdate \
    --data_file data/ppo/rp/ml-100k-reflection.jsonl \
    --output_file data/ppo/rp/ml-100k-v2.jsonl \
    --reward_version v2
```

**Step 3: RLHF Training**
```bash
# Train with PPO using generated feedback data
python main.py -m RLHFTraining \
    --config_path config/training/ppo-main.json \
    --epochs 1 \
    --data_file data/ppo/rp/ml-100k-v2.jsonl
```

**Step 4: Continued Training**
```bash
# Continue training from checkpoint
python main.py -m RLHFTraining \
    --config_path config/training/ppo-main.json \
    --epochs 1 \
    --data_file data/ppo/rp/ml-100k-v2.jsonl \
    --model_path ckpts/xxxx/epoch-0
```

#### 2. Feedback Generation Script (`/scripts/feedback.sh`)

**Rating Prediction Task**:
```bash
# Generate reflection feedback
python main.py --main Feedback --data_file data/ml-100k/train.csv --system reflection --task rp --feedback_file data/ppo/rp/ml-100k-reflection.jsonl --reward_version reflection

# Update to V2 rewards  
python main.py --main RewardUpdate --data_file data/ppo/rp/ml-100k-reflection.jsonl --output_file data/ppo/rp/ml-100k-v2.jsonl --reward_version v2
```

**Sequential Recommendation Task**:
```bash
# Generate reflection feedback
python main.py --main Feedback --data_file data/ml-100k/train.csv --system reflection --task sr --max_his 5 --feedback_file data/ppo/sr/ml-100k-reflection.jsonl --reward_version reflection

# Update to V1 rewards
python main.py --main RewardUpdate --data_file data/ppo/sr/ml-100k-reflection.jsonl --output_file data/ppo/sr/ml-100k-v1.jsonl --reward_version v1 --task sr
```

### Key Innovation: Multi-Agent Feedback Integration

#### 1. Agent-Based Reward Signal Generation

**Reflection Agent as Reward Source**:
- Generates self-assessment in JSON format: `{"correctness": bool, "reason": str}`
- Provides natural language explanations for decisions
- Enables reward functions to assess both accuracy and reasoning quality

**Multi-Agent Collaboration for Training**:
```python
# Reward calculation uses full agent interaction history
record['reward'] = self.reward_model(
    action1=record["Answer_1"],           # Initial agent response
    action2=record["Answer_2"],           # Response after agent reflection
    gt_answer=record["Answer_GT"],        # Ground truth
    reflection_output=record["output"]    # Agent's self-assessment
)
```

#### 2. Fine-Tuning Benefits

**Advantages over Standard Supervised Learning**:
1. **Reward-Driven Optimization**: Manager's LLMs directly optimize for recommendation quality metrics
2. **Multi-Turn Learning**: Learns from improvement between initial and reflected Manager responses
3. **Self-Assessment**: Incorporates Manager's ability to evaluate its own performance through reflection
4. **Task-Specific Rewards**: Different reward functions for different recommendation tasks

**Technical Implementation Details**:
- **LoRA Fine-Tuning**: Uses r=16, alpha=16 for efficient adaptation of Manager's LLMs only
- **PPO Training**: Learning rate 1.41e-5, batch size 2 for stable Manager training
- **Offline Training**: Uses pre-collected feedback data from multi-agent interactions rather than online exploration
- **Multi-Task Support**: Manager learns separate behaviors for RP and SR tasks through different reward functions

**What the Manager Learns**:
- Better thought processes for analyzing recommendation tasks
- More effective action generation (when to call which agents)
- Improved final answer synthesis based on agent responses
- Enhanced ability to leverage reflection feedback for quality improvement

### Fine-Tuning Outcomes

**Expected Improvements for Manager Agent**:
- **Better Calibration**: Manager learns to assess its own confidence in recommendations
- **Improved Reasoning**: Reflection-based training enhances Manager's explanatory capabilities
- **Task Adaptation**: Reward signals guide Manager toward task-specific objectives
- **Multi-Agent Orchestration**: Manager learns optimal patterns for calling and coordinating other agents
- **Enhanced Synthesis**: Better ability to combine information from Analyst, Searcher, and other agents

**System-Level Benefits**:
- **Improved Coordination**: Better Manager leads to more effective multi-agent collaboration
- **Task-Specific Optimization**: Different Manager behaviors for rating prediction vs. sequential recommendation
- **Quality Assurance**: Enhanced self-assessment capabilities improve overall system reliability

This comprehensive RLHF pipeline represents a sophisticated approach to fine-tuning the central orchestrator (Manager) in a multi-agent recommendation system, going beyond traditional supervised learning to incorporate multi-agent feedback and reward-driven optimization while keeping other agents unchanged.

## Conclusion

MACRec represents a significant advancement in applying multi-agent systems to recommendation tasks. Its modular architecture, comprehensive agent types, and flexible configuration system make it a powerful framework for building sophisticated recommendation systems that leverage the reasoning capabilities of Large Language Models.

The framework's strength lies in its ability to decompose complex recommendation tasks into specialized agent roles, enabling more targeted and effective problem-solving compared to single-agent approaches. The extensive prompt management system and standardized communication protocols ensure consistency and reliability across different recommendation scenarios.

**Key Contributions of MACRec's Fine-Tuning Approach**:
1. **RLHF Integration**: Novel application of PPO training for recommendation tasks using multi-agent feedback
2. **Reward Function Design**: Task-specific reward functions that capture recommendation quality metrics
3. **Reflection-Based Training**: Incorporates agent self-assessment as a training signal
4. **Multi-Turn Learning**: Learns from the improvement process between initial and reflected responses
5. **Scalable Pipeline**: Complete training infrastructure from data generation to model deployment

The combination of multi-agent collaboration and sophisticated fine-tuning makes MACRec a comprehensive solution for building next-generation recommendation systems that can reason, reflect, and improve their performance through structured feedback mechanisms.