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

**Code Example:**
```python
class Manager(Agent):
    def __init__(self, thought_config_path: str, action_config_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thought_llm = self.get_LLM(thought_config_path)
        self.action_llm = self.get_LLM(action_config_path)
        
    def forward(self, stage: str, *args, **kwargs) -> str:
        if stage == 'thought':
            return self._prompt_thought(**kwargs)
        elif stage == 'action':
            return self._prompt_action(**kwargs)
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

**Reflection Strategies:**
```python
class ReflectionStrategy(Enum):
    NONE = 'base'                                    # No reflection
    LAST_ATTEMPT = 'last_trial'                     # Store last attempt
    REFLEXION = 'reflection'                        # LLM-based reflection
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflection'  # Combined approach
```

**Location**: `/macrec/agents/reflector.py:20-106`

#### 5. Task Interpreter Agent (`/macrec/agents/interpreter.py`)
Translates conversational dialogs into structured recommendation tasks, especially crucial for conversational recommendation scenarios.

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