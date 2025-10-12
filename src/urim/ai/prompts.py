# SYSTEM PROMPTS
OUTPUT_JSON_SYSTEM = (
    "You are a JSON parser. Output only fully valid JSON objects. Do not output a list"
    " at the top level."
)
OUTPUT_FUNCTION_SYSTEM = (
    "You are a function writer. Only output fully valid Python functions that can be"
    " run with the following external dependencies:\n- pandas\n- numpy\nIf you need to"
    " use any external dependencies, import them at the start of your function."
)
COT_SYSTEM = (
    "You are a careful, intelligent assistant capable of reasoning through difficult problems"
    " step-by-step. Before you answer any question, first reason through the problem internally"
    " using a scratchpad to plan and verify your answer. You should take as much or as little time"
    " as you need to help you get to an answer you are confident in.\n\nWhen structuring your"
    " output, first do your thinking between a pair of XML <{tag}>...</{tag}> tags. This will be"
    " helpful with parsing your intermediate thinking from your final response later down the line."
    " Note that when thinking, anything in between these tags will not visible to the user, so you"
    " don't necessarily need to follow particular rules, such as those related to formatting, that"
    " a user may ask you to follow in your final response. Once you're done thinking and have"
    " closed off your thinking tags, respond to the user's question normally, following any"
    " additional guidelines they may set, and to the best of your abilities given the thinking you"
    " did internally."
)
COT_WITH_INSTRUCTIONS_SYSTEM = (
    COT_SYSTEM
    + "\n\nFinally, follow these instructions to structure your thinking:\n{instructions}"
)

# UTILITY PROMPTS
DATASET_RENAME_PROMPT = (
    "I have a dataset with the following columns: {columns}. Here is also the head"
    " output from the dataset:\n{head}\n\nI want to rename the columns with the"
    " following scheme: {scheme}. Please output a valid mapping from existing column"
    " names that need to be changed to new column names (as you would see in"
    " pd.DataFrame.rename)."
)
DATASET_DROP_NO_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. Here is also the head"
    " output from the dataset:\n{head}\n\nI want to drop some columns from the dataset."
    " Please output a JSON object with one key `columns` containing a list of column"
    " names to drop."
)
DATASET_DROP_WITH_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. Here is also the head"
    " output from the dataset:\n{head}\n\nI want to drop some columns from the dataset."
    " Please output a JSON object with one key `columns` containing a list of column"
    " names to drop. Here's a description of the desired dropping behavior:\n{scheme}"
)
DATASET_FILTER_PROMPT = (
    "I have a dataset with the following columns: {columns}. Here is also the head"
    " output from the dataset:\n{head}\n\nI want to filter the dataset based on a"
    " condition. Please write a Python function that takes as input a pandas Series and"
    " returns a boolean. This will be used as `fn` in `df.apply(fn, axis=1)` to filter"
    " the dataset. Here's a description of the desired filtering"
    " behavior:\n{scheme}\n\nWhen writing your function, don't use any type hints, if"
    " you need to import any external libraries, do so at the beginning of your"
    " response."
)
DATASET_APPLY_PROMPT = (
    "I have a dataset with the following columns: {columns}. Here is also the head"
    " output from that dataset:\n{head}\n\nI want to apply a function to each row of"
    " the dataset. Please write a python function that takes can be run without inputs"
    " that returns a length two tuple with the following values:\n1) a string"
    " representing the name for the new column. {column_hint}\n2) a function that takes"
    " as input a pandas Series and returns a value that will be mapped across each row"
    " of the original dataset to create the new column. This function will be used as"
    " `fn` in `df[new_col] = df.apply(fn, axis=1)` to apply the function to the"
    " dataset. Here's a description of the desired mapping behavior:\n{scheme}\n\nWhen"
    " writing your function, don't use any type hints, if you need to import any"
    " external libraries, do so at the beginning of your response."
)
DATASET_MERGE_NO_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. I want to join this"
    " dataset with another one with the following columns: {other_columns}. Here is the"
    " head output from the first dataset, which I'll refer to as `df1` from now"
    " on.\n{head}\n\nHere is the head output from the second dataset, which I'll refer"
    " to as `df2` from now on.\n{other_head}\n\nI will be doing this join via"
    " `pandas.merge`, and I need you to help me fill in some missing args to make the"
    " operation successful. For your response, please output a JSON with the following"
    " keys:\n1) one of `on`, `left_on`, or `right_on`\n2) `how` (defaults to `left` if"
    " not provided) These will be passed in as kwargs to the following expression: `df1"
    " = df1.merge(df2, **kwargs)`."
)
DATASET_MERGE_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. I want to join this"
    " dataset with another one with the following columns: {other_columns}. Here is the"
    " head output from the first dataset, which I'll refer to as `df1` from now"
    " on.\n{head}\n\nHere is the head output from the second dataset, which I'll refer"
    " to as `df2` from now on.\n{other_head}\n\nI will be doing this join via"
    " `pandas.merge`, and I need you to help me fill in some missing args to make the"
    " operation successful. Here's a description of the desired merging"
    " behavior:\n{scheme}\n\nFor your response, please output a JSON with the following"
    " keys:\n1) one of `on`, `left_on`, or `right_on`\n2) `how` (defaults to `left` if"
    " not provided)\nThis JSON will be passed in as kwargs to the following expression:"
    " `df1 = df1.merge(df2, **kwargs)`."
)
DATASET_CONCAT_NO_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. I want to concatenate this"
    " dataset with another one with the following columns: {other_columns}. Here is the"
    " head output from the first dataset, which I'll refer to as `df1` from now"
    " on.\n{head}\n\nHere is the head output from the second dataset, which I'll refer"
    " to as `df2` from now on.\n{other_head}\n\nI will be doing this concatenation via"
    " `pd.concat`. For your response, please output one JSON that contains a mapping"
    " for renaming columns from both `df1` and `df2` such that the concatenation makes"
    " the most sense. The keys that contain `df1` column names should be prepended with"
    " `df1_`, and similarly prepend `df2_` for `df2` columns.\n\nHere's an"
    " example:\nSay the user wants to concatenate a dataframe with the following"
    " columns: a, b with another dataframe with the following columns: d, e, f. The"
    " user wants to achieve the following goal: a should be concatenated with d, f"
    " should be concatenated with b. Your response should be the following"
    ' JSON:\n{{\n  "df2_d": "a",\n  "df2_f": "b"\n}}.\n\nKeep the following'
    " goals in mind:\n1) When you have the option of either renaming a column from"
    " `df1` or `df2`, elect to rename columns from `df2`.\n2) You do not need to ensure"
    " that the column names are completely identical across the two datasets, as this"
    " will sometimes be unrealistic. You should try to minimize the number of columns"
    " that need to be renamed, but merge as many columns as possible that seem to have"
    " reasonably similar semantic purpose or structure."
)
DATASET_CONCAT_HINT_PROMPT = (
    "I have a dataset with the following columns: {columns}. I want to concatenate this"
    " dataset with another one with the following columns: {other_columns}. Here is the"
    " head output from the first dataset, which I'll refer to as `df1` from now"
    " on.\n{head}\n\nHere is the head output from the second dataset, which I'll refer"
    " to as `df2` from now on.\n{other_head}\n\nI will be doing this concatenation via"
    " `pd.concat`. For your response, please output one JSON that contains a mapping"
    " for renaming columns from both `df1` and `df2` such that the concatenation makes"
    " the most sense. The keys that contain `df1` column names should be prepended with"
    " `df1_`, and similarly prepend `df2_` for `df2` columns.\n\nHere's an"
    " example:\nSay the user wants to concatenate a dataframe with the following"
    " columns: a, b with another dataframe with the following columns: d, e, f. The"
    " user wants to achieve the following goal: a should be concatenated with d, f"
    " should be concatenated with b. Your response should be the following"
    ' JSON:\n{{\n  "df2_d": "a",\n  "df2_f": "b"\n}}.\n\nKeep the following'
    " goals in mind:\n1) When you have the option of either renaming a column from"
    " `df1` or `df2`, elect to rename columns from `df2`.\n2) You do not need to ensure"
    " that the column names are completely identical across the two datasets, as this"
    " will sometimes be unrealistic. Instead, you should rename enough columns such"
    " that you achieve the following goal: {scheme}"
)
GENERATE_DESCRIBE_CHAIN_PROMPT = """\
I have a dataset with the following columns: {columns}. Additionally, here is the head output from the dataset:
{head}

I want to perform the following changes:
{scheme}

To assist me with this task, you have access to the functions that I've defined below. You will need to find a way to chain together a series of these function calls, where the output of one call is the input to the next, and the final output of the last call is the desired dataset.

Function definitions:
Name: sample
Description: Randomly sample a subset of the dataset.
Arguments:
  - n: int
    required: False
    description: Absolute number of rows to randomly sample from the dataset.
  - frac: float
    required: False
    description: Fraction of rows to sample randomly from the dataset.
Name: rename
Description: Rename the columns of the dataset.
Arguments:
  - hint: str
    required: True
    description: Natural language description of the desired renaming scheme.
Name: drop
Description: Drop columns from the dataset.
Arguments:
  - hint: str
    required: True
    description: Natural language description of the desired column dropping scheme.
Name: filter
Description: Filter the dataset based on a condition.
Arguments:
  - hint: str
    required: True
    description: Natural language description of the desired filtering scheme.
Name: apply
Description: Apply a function to each row of the dataset and output the results in a column.
Arguments:
  - column: str
    required: True
    description: The name of the column to output the results of the function to.
  - hint: str
    required: True
    description: Natural language description of the desired function that will be applied to each row of the dataset.

Some additional rules you should keep in mind:
1) When outputting your response, you should follow a very strict formatting. Each line should contain exactly one function call. For each function call, follow this syntax: [function name] | [keyword argument 1]=[value 1] | [keyword argument 2]=[value 2] | ... Everything wrapped in square braces should be replaced with actual values. You do not need to include quotes around your string values.
2) None of these functions take in the actual dataset as an argument. You can assume that they will all have access to the dataset via some global variable. These functions will be called in the order that you output them.
3) You should NEVER output any function calls that are not in the list of functions above.
"""
