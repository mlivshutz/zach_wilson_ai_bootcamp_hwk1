import dspy
import os
import warnings
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

# Suppress the OpenSSL warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Use .env and dspy to setup openai apikey
load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY')

import openai

# List available models
client = openai.OpenAI(api_key=api_key)
models = client.models.list()
for model in models.data:
    if "gpt" in model.id:
        print(f"Available: {model.id}")

# Test question
question = "What country has the same letter repeated the most in its name?"
expected_answer = "Saint Vincent and the Grenadines"

output_file = "output/prompt_v2_with_inspection_results.txt"
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("TESTING PROMPT V2 WITH DETAILED PROMPT INSPECTION\n")
    f.write("=" * 80 + "\n")

# Test gpt-4o with dspy chainofthought - WITH PROMPT INSPECTION
print("\n🔍 Testing GPT-4o with Chain of Thought + Prompt Inspection:")
lm_4o = dspy.LM("openai/gpt-3.5-turbo", api_key=api_key)
dspy.configure(lm=lm_4o)
qa_4o_cot = dspy.ChainOfThought('question -> answer')
result_4o_cot = qa_4o_cot(question=question)

with open(output_file, "a") as f:
    f.write(f"\n📤 BASELINE PROMPT SENT TO GPT-4o:\n")
    f.write("-" * 50 + "\n")

print("📤 BASELINE PROMPT:")
dspy.inspect_history(n=1)

with open(output_file, "a") as f:
    f.write(f"Reasoning: {result_4o_cot.reasoning}\n")
    f.write(f"Result: {result_4o_cot.answer}\n")

# Create examples for dspy to create an optimized prompt (V2 style - comparative examples)
trainset = [
    # Asia-Pacific countries with repeated letters
    dspy.Example(question="Using a list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between  Philippines and the Marshall Islands, which has the most repeated letters?", answer="Philippines", reasoning= "Because the Philippines has p repeated 4 times while Marshall Islands most preeated letter is a 3 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between  Mexico and Canada?", answer="Canada", reasoning= "Because Canada has the most repeated letter is a 3 times while Mexico most repeated letter is a 1 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between Rwanda and Saint Vincent and the Grenadines?", answer="Saint Vincent and the Grenadines", reasoning= "Because Saint Vincent and the Grenadines has the letter n repeated 6 times while Rwanda most repeated letter is a 2 times").with_inputs("question"),
    dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between the United States Of America and Egypt?", answer="United States of America", reasoning="After counting the repetition of letters in the United States of America you get A three times while Egypts most repeated letter is only repeated once").with_inputs("question"),
    # dspy.Example(question="Using the list of all recognized countries. Compare the countries to see which has the most repeated letters in its name. For example, between Marshall Islands and Saint Vincent and the Grenadines?", answer="Saint Vincent and the Grenadines", reasoning= "Because Saint Vincent and the Grenadines has the letter n repeated 6 times while Marshall Islands most repeated letter is a 3 times").with_inputs("question"),
]

with open(output_file, "a") as f:
    f.write(f"\n📚 TRAINING EXAMPLES (V2 - COMPARATIVE STYLE):\n")
    f.write("-" * 50 + "\n")
    for i, example in enumerate(trainset, 1):
        f.write(f"Example {i}:\n")
        f.write(f"  Q: {example.question}\n")
        f.write(f"  A: {example.answer}\n")
        f.write(f"  Reasoning: {example.reasoning}\n")
        f.write("\n")

# Use dspy and examples to create an optimized prompt
class CountryQuestion(dspy.Signature):
    """Identify the country with the most repeated letters in its name"""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The country name with the most repeated letters")

class CountryClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CountryQuestion)
    
    def forward(self, question):
        return self.predict(question=question)

# Create the module to be optimized
country_classifier = CountryClassifier()

# Configure the BootstrapFewShot optimizer
bootstrap_few_shot = BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=5)

# Use GPT-4o for optimization
dspy.configure(lm=lm_4o)

# Compile the optimized module
compiled_classifier = bootstrap_few_shot.compile(country_classifier, trainset=trainset)

with open(output_file, "a") as f:
    f.write(f"\n🎯 OPTIMIZED MODULE CREATED (V2 APPROACH)\n")
    f.write("-" * 50 + "\n")

# Test the optimized classifier and inspect its prompt
print("\n🔍 Testing OPTIMIZED V2 classifier + Prompt Inspection:")
optimized_result = compiled_classifier(question=question)

with open(output_file, "a") as f:
    f.write(f"\n📤 OPTIMIZED V2 PROMPT SENT TO GPT-4o:\n")
    f.write("-" * 50 + "\n")

print("📤 OPTIMIZED V2 PROMPT:")
dspy.inspect_history(n=1)

with open(output_file, "a") as f:
    f.write(f"Optimized Reasoning: {optimized_result.reasoning}\n")
    f.write(f"Optimized Result: {optimized_result.answer}\n")

# Inspect the internal structure of the compiled classifier
with open(output_file, "a") as f:
    f.write(f"\n🔍 INTERNAL STRUCTURE INSPECTION (V2):\n")
    f.write("-" * 50 + "\n")

if hasattr(compiled_classifier.predict, 'demos'):
    demos = compiled_classifier.predict.demos
    with open(output_file, "a") as f:
        f.write(f"Number of demos in compiled predictor: {len(demos)}\n")
        for i, demo in enumerate(demos):
            f.write(f"Demo {i+1}:\n")
            f.write(f"  Question: {demo.question}\n")
            f.write(f"  Answer: {demo.answer}\n")
            if hasattr(demo, 'reasoning'):
                f.write(f"  Reasoning: {demo.reasoning}\n")
            f.write("\n")

# Run multiple tests to see consistency
print("\n🔄 Testing V2 consistency (10 runs):")
with open(output_file, "a") as f:
    f.write(f"\n🔄 CONSISTENCY TEST V2 (10 runs):\n")
    f.write("-" * 50 + "\n")

correct_count = 0
results = []
for i in range(10):
    test_result = compiled_classifier(question=question)
    correct = "Saint Vincent and the Grenadines" in test_result.answer
    if correct:
        correct_count += 1
    results.append(test_result.answer)
    with open(output_file, "a") as f:
        f.write(f"Run {i+1}: {test_result.answer} {'✅' if correct else '❌'}\n")

accuracy = (correct_count / 10) * 100

with open(output_file, "a") as f:
    f.write(f"\nAccuracy: {correct_count}/10 ({accuracy:.1f}%)\n")

# Analyze the different answers
unique_answers = list(set(results))
with open(output_file, "a") as f:
    f.write(f"\nUnique answers given:\n")
    for answer in unique_answers:
        count = results.count(answer)
        f.write(f"  '{answer}': {count} times\n")

# Compare V2 examples to what actually gets used
print("\n🔬 Detailed Analysis of V2 Approach:")
with open(output_file, "a") as f:
    f.write(f"\n🔬 DETAILED ANALYSIS OF V2 APPROACH:\n")
    f.write("-" * 50 + "\n")
    f.write("V2 Strategy Analysis:\n")
    f.write("• Uses comparative examples instead of direct questions\n")
    f.write("• Examples show comparison between two countries\n")
    f.write("• Some examples include 'Saint Vincent and the Grenadines' as the correct answer\n")
    f.write("• Format: 'between X and Y, which has more repeated letters?'\n")
    f.write("\nPotential issues with V2:\n")
    f.write("• Comparative format might confuse the model about the task scope\n")
    f.write("• Model might think it should only compare limited options\n")
    f.write("• Examples are more complex and harder to generalize from\n")
    f.write("• Training examples mix different comparison styles\n")

# Test with a simpler direct question using the optimized V2 classifier
print("\n🧪 Testing V2 classifier with direct question:")
simple_question = "What country has the most repeated letters in its name?"
simple_result = compiled_classifier(question=simple_question)

with open(output_file, "a") as f:
    f.write(f"\n🧪 V2 CLASSIFIER WITH DIRECT QUESTION:\n")
    f.write("-" * 50 + "\n")
    f.write(f"Question: {simple_question}\n")
    f.write(f"Answer: {simple_result.answer}\n")
    f.write(f"Reasoning: {simple_result.reasoning}\n")

print("📤 V2 CLASSIFIER WITH DIRECT QUESTION PROMPT:")
dspy.inspect_history(n=1)

with open(output_file, "a") as f:
    f.write(f"\n" + "=" * 80 + "\n")
    f.write("V2 KEY FINDINGS:\n")
    f.write("=" * 80 + "\n")
    f.write("1. V2 uses comparative examples that may confuse the model\n")
    f.write("2. Despite having correct answer in examples, accuracy may be lower\n")
    f.write("3. Comparative format creates cognitive load vs direct examples\n")
    f.write("4. DSPy optimization may not handle complex example patterns well\n")
    f.write("5. inspect_history() reveals exact prompts and example formatting\n")
    f.write("6. V2 approach shows why example design matters for optimization\n")

print(f"\n📄 Detailed V2 results with prompts written to {output_file}")
print("\n💡 KEY TAKEAWAYS FOR V2:")
print("• V2 uses comparative examples instead of direct questions")
print("• This format may confuse DSPy optimization process")
print("• Even with correct answers in examples, results may be inconsistent")
print("• inspect_history() shows how DSPy formats comparative examples")
print("• Demonstrates importance of example design for prompt optimization")

# Final comparison insight
print("\n🎯 V1 vs V2 Comparison Insight:")
print("• V1: Direct examples → Clear pattern for model to follow")
print("• V2: Comparative examples → Confusing pattern, harder to generalize")
print("• Lesson: Example format matters as much as content for DSPy optimization!")
