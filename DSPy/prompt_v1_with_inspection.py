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

# Test question
question = "What country has the same letter repeated the most in its name?"
expected_answer = "Saint Vincent and the Grenadines"

output_file = "output/prompt_v1_with_inspection_results.txt"
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("TESTING PROMPT V1 WITH DETAILED PROMPT INSPECTION\n")
    f.write("=" * 80 + "\n")

# Test gpt-4o with dspy chainofthought - WITH PROMPT INSPECTION
print("\n🔍 Testing GPT-4o with Chain of Thought + Prompt Inspection:")
lm_4o = dspy.LM("openai/gpt-4o", api_key=api_key)
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

# Create examples for dspy to create an optimized prompt
trainset = [
    dspy.Example(question="What country has the same letter repeated the most in its name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="Which nation has the highest frequency of repeated letters in its name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="What is the country with the most letter repetitions in its official name?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="Which country's name contains the same letter appearing most frequently?", answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="What nation has a name with the highest count of duplicate letters?", answer="Saint Vincent and the Grenadines").with_inputs("question")
]

with open(output_file, "a") as f:
    f.write(f"\n📚 TRAINING EXAMPLES:\n")
    f.write("-" * 50 + "\n")
    for i, example in enumerate(trainset, 1):
        f.write(f"Example {i}: Q: {example.question}\n")
        f.write(f"            A: {example.answer}\n")

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
    f.write(f"\n🎯 OPTIMIZED MODULE CREATED\n")
    f.write("-" * 50 + "\n")

# Test the optimized classifier and inspect its prompt
print("\n🔍 Testing OPTIMIZED classifier + Prompt Inspection:")
optimized_result = compiled_classifier(question=question)

with open(output_file, "a") as f:
    f.write(f"\n📤 OPTIMIZED PROMPT SENT TO GPT-4o:\n")
    f.write("-" * 50 + "\n")

print("📤 OPTIMIZED PROMPT:")
dspy.inspect_history(n=1)

with open(output_file, "a") as f:
    f.write(f"Optimized Reasoning: {optimized_result.reasoning}\n")
    f.write(f"Optimized Result: {optimized_result.answer}\n")

# Inspect the internal structure of the compiled classifier
with open(output_file, "a") as f:
    f.write(f"\n🔍 INTERNAL STRUCTURE INSPECTION:\n")
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
print("\n🔄 Testing consistency (5 runs):")
with open(output_file, "a") as f:
    f.write(f"\n🔄 CONSISTENCY TEST (5 runs):\n")
    f.write("-" * 50 + "\n")

for i in range(5):
    test_result = compiled_classifier(question=question)
    correct = "Saint Vincent and the Grenadines" in test_result.answer
    with open(output_file, "a") as f:
        f.write(f"Run {i+1}: {test_result.answer} {'✅' if correct else '❌'}\n")

with open(output_file, "a") as f:
    f.write(f"\n" + "=" * 80 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("=" * 80 + "\n")
    f.write("1. Baseline prompt: Simple question-answer format\n")
    f.write("2. Optimized prompt: Includes few-shot examples from training\n")
    f.write("3. DSPy automatically formats examples into the prompt\n")
    f.write("4. inspect_history() shows exactly what was sent to the LLM\n")
    f.write("5. Compiled modules store examples in .predict.demos\n")

print(f"\n📄 Detailed results with prompts written to {output_file}")
print("\n💡 KEY TAKEAWAYS:")
print("• dspy.inspect_history(n=1) shows the EXACT prompt sent to LLM")
print("• Optimized prompts include your training examples as few-shot examples")
print("• You can see internal demos with: compiled_classifier.predict.demos")
print("• This reveals WHY certain approaches work better than others!")
