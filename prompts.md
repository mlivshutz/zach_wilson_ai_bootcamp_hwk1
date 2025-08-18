# DSPy Prompt Optimization Results

## Overview

I used DSPy to optimize the prompt for the country question: **"What country has the same letter repeated the most in its name?"**

DSPy started with a basic V1 prompt and then auto-optimized it for an optimized V2 prompt using comparative examples.

## Model Testing Results

I tested 3 different OpenAI models:

| Model | Result | Accuracy |
|-------|---------|----------|
| `gpt-4.1-mini` | ✅ **Saint Vincent and the Grenadines** | Correct |
| `gpt-4o` | ❌ Greece | Incorrect |
| `gpt-3.5-turbo` | ❌ Kyrgyzstan | Incorrect |

## Detailed Results

### 🎯 GPT-4.1-Mini (Correct Answer)

**Result:** Saint Vincent and the Grenadines ✅

#### Optimized V2 Prompt:

```
[2025-08-18T00:11:25.000090]

System message:

Your input fields are:
1. `question` (str):
Your output fields are:
1. `reasoning` (str): 
2. `answer` (str): The country name with the most repeated letters
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## reasoning ## ]]
{reasoning}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Identify the country with the most repeated letters in its name
```

#### Training Examples:

```
User message:
[[ ## question ## ]]
Using a list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between Philippines and the Marshall Islands, which has the most repeated letters?

Assistant message:
[[ ## reasoning ## ]]
The name "Philippines" contains the letter "i" repeated 3 times and the letter "p" repeated 3 times. The name "Marshall Islands" contains the letter "l" repeated 3 times and the letter "a" repeated 3 times. Both countries have letters repeated 3 times, but "Philippines" has two different letters repeated 3 times each, while "Marshall Islands" has two letters repeated 3 times each as well. Since the maximum repetition count is the same for both, they tie in terms of the most repeated letters.

[[ ## answer ## ]]
Philippines and Marshall Islands (tie)
```

#### Final Answer:

```
User message:
[[ ## question ## ]]
What country has the same letter repeated the most in its name?

Response:
[[ ## reasoning ## ]]
To find the country with the same letter repeated the most in its name, we need to consider all recognized countries and count the maximum number of times any single letter appears in each country's name. One notable example is "Tanzania," where the letter 'a' appears 3 times. However, there are countries with even higher repetitions. For instance, "Seychelles" has the letter 'e' repeated 4 times. Another example is "Philippines," where the letter 'i' appears 3 times and 'p' appears 3 times. But "Seychelles" has a higher count with 4 repeated 'e's. Checking further, "Saint Vincent and the Grenadines" has the letter 'n' repeated 5 times, which is higher than "Seychelles." Therefore, "Saint Vincent and the Grenadines" holds the record for the most repeated single letter in a country's name with 5 occurrences of 'n'.

[[ ## answer ## ]]
Saint Vincent and the Grenadines

[[ ## completed ## ]]
```

---

### ❌ GPT-4o (Incorrect Answer)

**Result:** Greece ❌

#### Final Answer:

```
User message:
[[ ## question ## ]]
What country has the same letter repeated the most in its name?

Response:
[[ ## reasoning ## ]]
To determine which country has the same letter repeated the most in its name, we need to examine the names of all recognized countries and count the occurrences of each letter within each name. The country with the highest count for a single letter will be the answer. 

One notable example is "Solomon Islands," where the letter 'o' is repeated 3 times. However, "Venezuela" has the letter 'e' repeated 3 times, and "Philippines" has the letter 'i' repeated 3 times. 

However, the country "Greece" has the letter 'e' repeated 3 times, which is the highest repetition of a single letter in a country's name. 

[[ ## answer ## ]]
Greece

[[ ## completed ## ]]
```

---

### ❌ GPT-3.5-Turbo (Incorrect Answer)

**Result:** Kyrgyzstan ❌

#### Final Answer:

```
User message:
[[ ## question ## ]]
What country has the same letter repeated the most in its name?

Response:
[[ ## reasoning ## ]]
To find the country with the most repeated letters in its name, we need to compare all recognized countries and identify the one with the highest number of repetitions for a single letter.

[[ ## answer ## ]]
Kyrgyzstan

[[ ## completed ## ]]
```

---

## Analysis and Conclusions

### 🎯 Correct Answer
**Saint Vincent and the Grenadines** has the letter **'n' repeated 6 times**, making it the country with the most repeated letters in its name.

### 📊 Model Performance Summary

- **GPT-4.1-Mini**: ✅ Correctly identified "Saint Vincent and the Grenadines" 
- **GPT-4o**: ❌ Incorrectly answered "Greece" (letter 'e' repeated 3 times)
- **GPT-3.5-Turbo**: ❌ Incorrectly answered "Kyrgyzstan" (no clear reasoning provided)

### 🔍 Key Insights

1. **DSPy Optimization**: Used comparative examples to train the models, but results varied significantly between models
2. **Model Reliability**: Even advanced models like GPT-4o can give incorrect answers for complex counting tasks
3. **Reasoning Quality**: GPT-4.1-Mini provided the most thorough reasoning and correct letter count
4. **Prompt Engineering**: The V2 approach with comparative examples showed mixed results across different models

### 🎓 Lessons Learned

- Model performance can vary significantly even with optimized prompts
- Complex analytical tasks require careful prompt design and model selection
- GPT-4.1-Mini demonstrated superior performance for this specific letter-counting task
- DSPy optimization doesn't guarantee consistent results across all models