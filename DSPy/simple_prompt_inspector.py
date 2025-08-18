#!/usr/bin/env python3
"""
Simple DSPy Prompt Inspector - Shows exactly what prompts DSPy generates

Run this to see the actual prompts DSPy creates during optimization.
"""

import dspy
import os
from dotenv import load_dotenv

# Setup
load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM("openai/gpt-4o", api_key=api_key)
dspy.configure(lm=lm)

question = "What country has the same letter repeated the most in its name?"

print("🔍 DSPy Prompt Inspector")
print("="*50)

# Method 1: Most Important - inspect_history()
print("\n1️⃣  BASIC CHAIN OF THOUGHT PROMPT")
print("-"*30)

cot = dspy.ChainOfThought('question -> answer')
result = cot(question=question)

print("📤 GENERATED PROMPT:")
dspy.inspect_history(n=1)
print(f"\n📥 RESULT: {result.answer}")

# Method 2: See optimized prompts with examples
print("\n\n2️⃣  OPTIMIZED PROMPT WITH EXAMPLES")
print("-"*30)

# Quick training set
trainset = [
    dspy.Example(question="What country has the most repeated letters?", 
                answer="Saint Vincent and the Grenadines").with_inputs("question"),
]

# Create optimized module
from dspy.teleprompt import BootstrapFewShot

class CountryClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought('question -> answer')
    
    def forward(self, question):
        return self.cot(question=question)

classifier = CountryClassifier()
optimizer = BootstrapFewShot(max_bootstrapped_demos=2)
compiled = optimizer.compile(classifier, trainset=trainset)

print("📤 OPTIMIZED PROMPT:")
opt_result = compiled(question=question)
dspy.inspect_history(n=1)
print(f"\n📥 OPTIMIZED RESULT: {opt_result.answer}")

# Method 3: Access internal demos
print("\n\n3️⃣  INTERNAL EXAMPLES/DEMOS")
print("-"*30)

if hasattr(compiled.cot, 'demos'):
    print(f"📊 Found {len(compiled.cot.demos)} demos in optimized module:")
    for i, demo in enumerate(compiled.cot.demos):
        print(f"  Demo {i+1}:")
        print(f"    Q: {demo.question}")
        print(f"    A: {demo.answer}")
        if hasattr(demo, 'reasoning'):
            print(f"    Reasoning: {demo.reasoning}")
else:
    print("ℹ️  No demos found in this compiled module")

print("\n\n💡 KEY INSIGHTS:")
print("-"*30)
print("• Use dspy.inspect_history(n=1) after ANY dspy call to see the prompt")
print("• Optimized modules include few-shot examples in the prompt")
print("• Chain-of-Thought adds reasoning instructions to prompts")
print("• compiled_module.predict.demos shows the examples DSPy learned")

print("\n🎯 To debug your V1/V2 results:")
print("-"*30)
print("1. Add 'dspy.inspect_history(n=1)' after each model call")
print("2. Look at compiled_classifier.predict.demos to see examples")
print("3. Compare the prompts between V1 and V2 to see differences")
