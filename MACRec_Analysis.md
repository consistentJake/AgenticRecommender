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

## Conclusion

MACRec represents a significant advancement in applying multi-agent systems to recommendation tasks. Its modular architecture, comprehensive agent types, and flexible configuration system make it a powerful framework for building sophisticated recommendation systems that leverage the reasoning capabilities of Large Language Models.

The framework's strength lies in its ability to decompose complex recommendation tasks into specialized agent roles, enabling more targeted and effective problem-solving compared to single-agent approaches. The extensive prompt management system and standardized communication protocols ensure consistency and reliability across different recommendation scenarios.