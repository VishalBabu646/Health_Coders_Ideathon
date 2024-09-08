import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize model and tokenizer
model = None
tokenizer = None
generator = None

def load_model(model_name):
    global model, tokenizer, generator

    print(f"Loading {model_name}...")

    # Configure GPU count
    gpu_count = torch.cuda.device_count()
    print('GPU count:', gpu_count)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Adjust according to your hardware capabilities
        low_cpu_mem_usage=True,
        cache_dir="cache"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Set the generator method
    generator = model.generate

# Use a valid model identifier
load_model("gpt2")

# Initial chat message
First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
print(First_chat)
history = [First_chat]

def go():
    invitation = "ChatDoctor: "
    human_invitation = "Patient: "

    # Get user input
    msg = input(human_invitation)
    print("")

    history.append(f"{human_invitation}{msg}")

    fulltext = f"If you are a doctor, please answer the medical questions based on the patient's description.\n\n{''.join(history)}\n\n{invitation}"

    # Tokenize and generate response
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,
            temperature=0.5,
            top_k=50,
            top_p=1.0,
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract response
    text_without_prompt = generated_text[len(fulltext):].strip()
    response = text_without_prompt.split(human_invitation)[0].strip()

    print(f"{invitation}{response}\n")
    history.append(f"{invitation}{response}")

while True:
    go()
