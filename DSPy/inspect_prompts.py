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

# Setup models
lm_4o = dspy.LM("openai/gpt-4o", api_key=api_key)
dspy.configure(lm=lm_4o)

# Test question
question = "What country has the same letter repeated the most in its name?"

print("="*80)
print("METHODS TO INSPECT DSPY PROMPTS")
print("="*80)

# Method 1: Use dspy.inspect_history() after any prediction
print("\n1. Using dspy.inspect_history() after a simple prediction:")
print("-" * 60)

qa_simple = dspy.Predict('question -> answer')
result = qa_simple(question=question)
print(f"Result: {result.answer}")
print("\nInspecting last interaction:")
dspy.inspect_history(n=1)

# Method 2: Inspect Chain of Thought prompts
print("\n\n2. Using dspy.inspect_history() after Chain of Thought:")
print("-" * 60)

qa_cot = dspy.ChainOfThought('question -> answer')
result_cot = qa_cot(question=question)
print(f"Result: {result_cot.answer}")
print("\nInspecting Chain of Thought interaction:")
dspy.inspect_history(n=1)

# Method 3: Create and inspect optimized module
print("\n\n3. Inspecting optimized module prompts:")
print("-" * 60)

# Create training examples (using V1 style for simplicity)
trainset = [
    dspy.Example(question="What country has the same letter repeated the most in its name?", 
                answer="Saint Vincent and the Grenadines").with_inputs("question"),
    dspy.Example(question="Which nation has the highest frequency of repeated letters in its name?", 
                answer="Saint Vincent and the Grenadines").with_inputs("question"),
]

# Define signature and module
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

# Create and compile the module
country_classifier = CountryClassifier()
bootstrap_few_shot = BootstrapFewShot(metric=None, max_bootstrapped_demos=3, max_labeled_demos=2)
compiled_classifier = bootstrap_few_shot.compile(country_classifier, trainset=trainset)

print("Running optimized classifier...")
optimized_result = compiled_classifier(question=question)
print(f"Optimized result: {optimized_result.answer}")

print("\nInspecting optimized module interaction:")
dspy.inspect_history(n=1)

# Method 4: Check the compiled module's internal structure
print("\n\n4. Inspecting compiled module structure:")
print("-" * 60)

# Access the internal predictor
internal_predictor = compiled_classifier.predict
print(f"Compiled module type: {type(compiled_classifier)}")
print(f"Internal predictor type: {type(internal_predictor)}")

# Try to access demos/examples if available
if hasattr(internal_predictor, 'demos'):
    print(f"\nNumber of demos in compiled predictor: {len(internal_predictor.demos)}")
    for i, demo in enumerate(internal_predictor.demos):
        print(f"Demo {i+1}:")
        print(f"  Question: {demo.question}")
        print(f"  Answer: {demo.answer}")
        if hasattr(demo, 'reasoning'):
            print(f"  Reasoning: {demo.reasoning}")

# Method 5: Enable verbose logging to see all interactions
print("\n\n5. Using DSPy settings for more detailed output:")
print("-" * 60)

# Configure DSPy for more verbose output
dspy.configure(lm=lm_4o, trace=[])

print("Running with trace enabled...")
traced_result = compiled_classifier(question=question)
print(f"Traced result: {traced_result.answer}")

# Show complete history
print("\nComplete interaction history:")
dspy.inspect_history(n=5)

print("\n" + "="*80)
print("ADDITIONAL INSPECTION METHODS")
print("="*80)

# Method 6: Manual inspection of what DSPy sends to the LM
print("\n6. Inspect raw LM calls:")
print("-" * 40)

# Create a custom LM wrapper to log all calls
class LoggingLM(dspy.LM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.call_count = 0
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1
        print(f"\n--- LM Call #{self.call_count} ---")
        if prompt:
            print(f"PROMPT:\n{prompt}")
        if messages:
            print(f"MESSAGES:")
            for msg in messages:
                print(f"  {msg}")
        print("--- End LM Call ---\n")
        
        # Call the parent method
        return super().__call__(prompt=prompt, messages=messages, **kwargs)

# Use the logging LM
logging_lm = LoggingLM("openai/gpt-4o", api_key=api_key)
dspy.configure(lm=logging_lm)

print("Running with logging LM to see exact prompts...")
qa_logged = dspy.ChainOfThought('question -> answer')
logged_result = qa_logged(question=question)
print(f"Logged result: {logged_result.answer}")

print("\n" + "="*80)
print("SUMMARY OF INSPECTION METHODS")
print("="*80)
print("1. dspy.inspect_history(n=X) - Shows last X interactions")
print("2. Enable tracing: dspy.configure(lm=lm, trace=[])")
print("3. Access demos: compiled_module.predict.demos")
print("4. Custom LM wrapper to log all prompts")
print("5. Save compiled modules and inspect their structure")
print("="*80)
