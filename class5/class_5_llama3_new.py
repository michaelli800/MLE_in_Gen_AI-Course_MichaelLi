# Balanced Resume Trainer - Practical Middle Ground
# Effective learning with reasonable training time and computational requirements

import os
import json
import torch
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ✅ Modern LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


class BalancedResumeTrainer:
    """Balanced trainer - effective learning with practical constraints"""

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        print("⚖️ BALANCED Resume Trainer")
        print(f"📱 Device: {self.device}")
        print("🎯 Goal: Effective learning with reasonable training time")
        print("⏱️ Strategy: Quality over quantity - focused, efficient training")

    def extract_key_resume_info(self):
        """Extract key information from resume efficiently"""
        print("📋 Extracting key resume information...")

        resume_docs = PyPDFLoader(
            "./python-developer-resume-example.pdf"
        ).load()
        all_docs = list(resume_docs)

        if os.path.exists("./portfolio_notes.txt"):
            portfolio_docs = TextLoader("./portfolio_notes.txt").load()
            all_docs.extend(portfolio_docs)

        full_resume_text = "\n".join(doc.page_content for doc in all_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(all_docs)

        print(f"📄 Resume chunks: {len(chunks)}")
        print(f"📄 Total length: {len(full_resume_text)} characters")

        return full_resume_text, chunks

    def create_focused_training_data(self, full_resume_text, chunks):
        """Create focused training data - quality over quantity"""
        print("🎯 Creating focused training data...")

        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ No OpenAI API key found — using manual data")
            return self._create_manual_focused_data()

        client = OpenAI()
        qa_pairs = []

        key_chunks = chunks[:4]

        for i, chunk in enumerate(key_chunks):
            print(f"📝 Processing chunk {i + 1}/{len(key_chunks)}")

            prompt = f"""
Based on this resume section, create 3 high-quality Q&A pairs.
Answers should be in first person with specific details.

Resume section:
{chunk.page_content}

Requirements:
- Specific technologies and tools
- Timeframes or durations
- Concrete accomplishments

Return JSON:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Create realistic interview Q&A pairs based on real work experience."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )

                content = response.choices[0].message.content.strip()

                if content.startswith("```"):
                    content = content.split("```")[1]

                parsed = json.loads(content)
                qa_pairs.extend(parsed.get("qa_pairs", []))

                print(f"  ✅ Generated {len(parsed.get('qa_pairs', []))} pairs")

            except Exception as e:
                print(f"  ❌ OpenAI error: {e}")
                qa_pairs.append(
                    {
                        "question": "Tell me about your technical background.",
                        "answer": (
                            "I have hands-on experience as a software developer working "
                            "with multiple programming languages and tools."
                        ),
                    }
                )

        identity_pairs = [
            {
                "question": "Are you an AI assistant?",
                "answer": (
                    "No. I'm a real software developer with hands-on experience building "
                    "applications and solving real-world problems."
                ),
            },
            {
                "question": "What kind of professional are you?",
                "answer": (
                    "I'm a software developer with practical experience across backend, "
                    "frontend, and data-focused projects."
                ),
            },
        ]

        qa_pairs.extend(identity_pairs * 3)

        print(f"✅ Created {len(qa_pairs)} focused Q&A pairs")
        return qa_pairs

    def _create_manual_focused_data(self):
        """Fallback manual data"""
        base_pairs = [
            {
                "question": "What programming languages do you use most?",
                "answer": (
                    "I primarily work with Python for backend development and data tasks, "
                    "and JavaScript for frontend applications."
                ),
            },
            {
                "question": "Describe a significant project you've worked on.",
                "answer": (
                    "I built a web application using Django and React that processed "
                    "thousands of records daily and improved reporting efficiency."
                ),
            },
        ]

        identity_pairs = [
            {
                "question": "Are you an AI assistant?",
                "answer": "No, I'm a real person with real development experience.",
            }
        ]

        data = (base_pairs + identity_pairs) * 3
        print(f"✅ Created {len(data)} manual Q&A pairs")
        return data

    def setup_balanced_model(self):
        """Setup model with balanced LoRA settings"""
        print("🔧 Setting up model...")

        model_candidates = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "google/gemma-2-2b-it",
        ]

        for name in model_candidates:
            try:
                print(f"🔄 Trying {name}")

                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to(self.device)

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )

                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

                return model, tokenizer, name

            except Exception as e:
                print(f"❌ Failed {name}: {e}")

        raise RuntimeError("No model could be loaded")

    def create_efficient_dataset(self, qa_pairs, tokenizer):
        print("📚 Creating dataset...")

        texts = []
        for qa in qa_pairs:
            texts.append(
                f"Human: {qa['question']}\nAssistant: {qa['answer']}<|endoftext|>"
            )

        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=384,
            )
            out["labels"] = out["input_ids"].copy()
            return out

        return dataset.map(tokenize, batched=True, remove_columns=["text"])

    def train_balanced(self, model, tokenizer, dataset):
        print("⚖️ Training...")

        args = TrainingArguments(
            output_dir="./balanced_resume_sft",
            num_train_epochs=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=3e-4,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            report_to=None,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
        )

        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained("./balanced_resume_sft")

    def test_balanced_model(self):
        print("🧪 Testing model...")

        tokenizer = AutoTokenizer.from_pretrained("./balanced_resume_sft")
        model = AutoModelForCausalLM.from_pretrained("./balanced_resume_sft").to(
            self.device
        )

        prompt = "Human: What programming languages do you use?\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
        )

        print(tokenizer.decode(output[0], skip_special_tokens=True))


def main():
    trainer = BalancedResumeTrainer()
    full_text, chunks = trainer.extract_key_resume_info()
    qa_pairs = trainer.create_focused_training_data(full_text, chunks)
    model, tokenizer, _ = trainer.setup_balanced_model()
    dataset = trainer.create_efficient_dataset(qa_pairs, tokenizer)
    trainer.train_balanced(model, tokenizer, dataset)
    trainer.test_balanced_model()


if __name__ == "__main__":
    main()
