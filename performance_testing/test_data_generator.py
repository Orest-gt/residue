#!/usr/bin/env python3
"""
PROJECT RESIDUE - Realistic Test Data Generator
=============================================

Generate realistic test data for various LLM use cases:
- Academic papers and research text
- Natural conversations and dialogue
- Code snippets and programming text
- Mixed content scenarios
"""

import sys
import random
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class TestDataGenerator:
    """Generate realistic test data for LLM performance testing"""

    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Academic vocabulary
        self.academic_terms = [
            "algorithm", "methodology", "hypothesis", "empirical", "statistical",
            "neural", "optimization", "convergence", "gradient", "backpropagation",
            "classification", "regression", "clustering", "dimensionality",
            "architecture", "framework", "implementation", "evaluation", "benchmark",
            "performance", "efficiency", "scalability", "robustness", "generalization"
        ]

        # Programming keywords
        self.code_keywords = [
            "def", "class", "import", "from", "return", "if", "else", "for", "while",
            "try", "except", "with", "lambda", "async", "await", "yield",
            "numpy", "torch", "tensorflow", "pandas", "sklearn", "matplotlib",
            "function", "variable", "parameter", "method", "attribute", "inheritance"
        ]

        # Conversation patterns
        self.conversation_starters = [
            "What do you think about", "Have you tried", "I'm wondering if",
            "Can you explain", "Do you know how", "What's your opinion on",
            "Have you noticed", "I'm curious about", "Can you help me with"
        ]

        print("=== TEST DATA GENERATOR ===")
        print("Ready to generate realistic test data")
        print("=" * 50)

    def generate_academic_text(self, length=500):
        """Generate academic/research-style text"""

        sentences = []

        # Introduction sentence
        sentences.append(f"The {random.choice(self.academic_terms)} approach presents a novel {random.choice(self.academic_terms)} for addressing complex {random.choice(self.academic_terms)} challenges.")

        # Body sentences
        for _ in range(length // 50 - 2):
            templates = [
                f"Our {random.choice(self.academic_terms)} demonstrates significant improvements in {random.choice(self.academic_terms)} performance.",
                f"The {random.choice(self.academic_terms)} results indicate a {random.randint(20,80)}% reduction in computational complexity.",
                f"Furthermore, the {random.choice(self.academic_terms)} exhibits robust {random.choice(self.academic_terms)} properties.",
                f"Comparative analysis with existing {random.choice(self.academic_terms)} methods reveals superior {random.choice(self.academic_terms)}.",
                f"The {random.choice(self.academic_terms)} framework enables efficient {random.choice(self.academic_terms)} processing."
            ]
            sentences.append(random.choice(templates))

        # Conclusion
        sentences.append(f"These findings establish the {random.choice(self.academic_terms)} as a promising direction for future research.")

        return " ".join(sentences)

    def generate_conversation_text(self, length=300):
        """Generate natural conversation-style text"""

        conversation_parts = []

        speakers = ["Alex", "Jordan", "Taylor", "Morgan", "Casey"]

        for i in range(length // 30):
            speaker = random.choice(speakers)
            starter = random.choice(self.conversation_starters)

            responses = [
                f"{starter} the latest developments in artificial intelligence?",
                f"{starter} machine learning has changed over the past few years?",
                f"{starter} the impact of neural networks on modern computing?",
                f"{starter} optimization algorithms work in practice?",
                f"{starter} the future of automated systems looks like?",
                f"{starter} we can improve computational efficiency in AI models?"
            ]

            conversation_parts.append(f"{speaker}: {random.choice(responses)}")

            # Add some follow-up
            if random.random() > 0.7:
                follow_ups = [
                    "That's really interesting. Can you elaborate?",
                    "I've been thinking about that too.",
                    "What are the main challenges there?",
                    "Have you seen any good examples?",
                    "That makes a lot of sense."
                ]
                conversation_parts.append(f"{random.choice(speakers)}: {random.choice(follow_ups)}")

        return "\n".join(conversation_parts)

    def generate_code_text(self, length=400):
        """Generate programming code-style text"""

        code_lines = []

        # Python function definitions
        functions = [
            "def optimize_model(parameters):",
            "def train_neural_network(data):",
            "def evaluate_performance(model):",
            "def preprocess_data(dataset):",
            "def visualize_results(metrics):",
            "class NeuralNetwork:",
            "def __init__(self, layers):",
            "def forward(self, input_data):",
            "def backward(self, gradients):",
            "def update_weights(self, learning_rate):"
        ]

        # Generate code structure
        for _ in range(length // 40):
            # Function/class definition
            code_lines.append(random.choice(functions))

            # Indented code block
            for _ in range(random.randint(3, 8)):
                indent = "    "
                code_patterns = [
                    f"{indent}{random.choice(self.code_keywords)} = {random.choice(['True', 'False', 'None', '0', '1'])}",
                    f"{indent}for i in range({random.randint(1,100)}):",
                    f"{indent}if {random.choice(self.code_keywords)} > {random.randint(0,10)}:",
                    f"{indent}return {random.choice(self.code_keywords)}",
                    f"{indent}print(f\"{random.choice(self.academic_terms)}: {{value}}\")",
                    f"{indent}# {random.choice(['Optimize', 'Process', 'Calculate', 'Update'])} {random.choice(self.code_keywords)}",
                    f"{indent}result = {random.choice(self.code_keywords)} * {random.randint(1,5)}",
                    f"{indent}data = np.{random.choice(['array', 'zeros', 'ones', 'random.randn'])}({random.randint(10,100)})"
                ]
                code_lines.append(random.choice(code_patterns))

            code_lines.append("")  # Empty line between blocks

        return "\n".join(code_lines)

    def generate_mixed_content(self, length=600):
        """Generate mixed content combining different text types"""

        sections = []
        remaining_length = length

        # Academic section
        academic_len = min(remaining_length // 3, 200)
        sections.append(f"## Research Overview\n\n{self.generate_academic_text(academic_len)}")
        remaining_length -= academic_len

        # Conversation section
        if remaining_length > 100:
            conv_len = min(remaining_length // 2, 150)
            sections.append(f"\n## Discussion\n\n{self.generate_conversation_text(conv_len)}")
            remaining_length -= conv_len

        # Code section
        if remaining_length > 50:
            code_len = remaining_length
            sections.append(f"\n## Implementation\n\n```python\n{self.generate_code_text(code_len)}\n```")

        return "\n".join(sections)

    def convert_text_to_numeric_data(self, text, method="token_counts"):
        """Convert text to numeric data for PROJECT RESIDUE testing"""

        if method == "token_counts":
            # Simple token counting approach
            words = text.split()
            # Convert words to numeric values (word length, position, etc.)
            numeric_data = []

            for i, word in enumerate(words[:1000]):  # Limit to 1000 tokens
                # Create features: word length, position, character variety
                features = [
                    len(word),  # Word length
                    i % 10,     # Position modulo 10
                    len(set(word.lower())),  # Unique characters
                    sum(1 for c in word if c.isupper()),  # Uppercase count
                    word.count('a') + word.count('e') + word.count('i') + word.count('o') + word.count('u')  # Vowels
                ]
                numeric_data.extend(features)

            return np.array(numeric_data, dtype=np.float32)

        elif method == "character_codes":
            # Character ASCII codes
            char_codes = [ord(c) for c in text[:1000]]
            return np.array(char_codes, dtype=np.float32)

        elif method == "word_frequencies":
            # Word frequency analysis
            words = text.lower().split()
            unique_words = list(set(words))
            frequencies = [words.count(word) for word in unique_words[:50]]  # Top 50 words
            return np.array(frequencies, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def generate_test_dataset(self, num_samples=50):
        """Generate comprehensive test dataset"""

        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_samples": num_samples,
                "generator_version": "1.0"
            },
            "samples": []
        }

        print(f"\nGenerating {num_samples} test samples...")

        content_types = ["academic", "conversation", "code", "mixed"]
        lengths = [200, 400, 600, 800]

        for i in range(num_samples):
            # Random content type and length
            content_type = random.choice(content_types)
            length = random.choice(lengths)

            # Generate text
            if content_type == "academic":
                text = self.generate_academic_text(length)
            elif content_type == "conversation":
                text = self.generate_conversation_text(length)
            elif content_type == "code":
                text = self.generate_code_text(length)
            else:
                text = self.generate_mixed_content(length)

            # Convert to numeric data
            numeric_data = self.convert_text_to_numeric_data(text)

            sample = {
                "id": i + 1,
                "content_type": content_type,
                "text_length": len(text),
                "numeric_length": len(numeric_data),
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "numeric_data": numeric_data.tolist()[:20] + ["..."] if len(numeric_data) > 20 else numeric_data.tolist()
            }

            dataset["samples"].append(sample)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

        return dataset

    def save_dataset(self, dataset, filename=None):
        """Save generated dataset to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_dataset_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"✅ Dataset saved to: {output_file}")
        return output_file

    def create_sample_files(self):
        """Create individual sample files for different content types"""

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        print("\nCreating individual sample files...")

        # Academic samples
        for i in range(5):
            text = self.generate_academic_text(400)
            numeric_data = self.convert_text_to_numeric_data(text)

            sample_file = samples_dir / f"academic_sample_{i+1}.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "academic",
                    "text": text,
                    "numeric_data": numeric_data.tolist()
                }, f, indent=2, ensure_ascii=False)

        # Conversation samples
        for i in range(5):
            text = self.generate_conversation_text(300)
            numeric_data = self.convert_text_to_numeric_data(text)

            sample_file = samples_dir / f"conversation_sample_{i+1}.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "conversation",
                    "text": text,
                    "numeric_data": numeric_data.tolist()
                }, f, indent=2, ensure_ascii=False)

        # Code samples
        for i in range(5):
            text = self.generate_code_text(350)
            numeric_data = self.convert_text_to_numeric_data(text)

            sample_file = samples_dir / f"code_sample_{i+1}.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "code",
                    "text": text,
                    "numeric_data": numeric_data.tolist()
                }, f, indent=2, ensure_ascii=False)

        print(f"✅ Created 15 individual sample files in {samples_dir}")

def main():
    """Main data generation function"""

    generator = TestDataGenerator()

    # Generate comprehensive dataset
    dataset = generator.generate_test_dataset(100)
    generator.save_dataset(dataset)

    # Create individual sample files
    generator.create_sample_files()

    print("\n✅ Test data generation completed!")
    print("📊 Generated realistic data for:")
    print("   - Academic research text")
    print("   - Natural conversations")
    print("   - Programming code")
    print("   - Mixed content scenarios")
    print("   - Numeric representations for PROJECT RESIDUE")

if __name__ == "__main__":
    main()
