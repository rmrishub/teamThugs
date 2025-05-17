import google.generativeai as genai

API_KEY = "AIzaSyCmeWKwXmyzqhOthmcg9oy6U7KfVmYVwnk"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

print("Chat with me! Type 'exit' to quit.")

# BERT-related code just defined but NOT called
def bert_code_for_display_only():
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    # Load your fine-tuned model and tokenizer (example paths)
    tokenizer = BertTokenizer.from_pretrained("./saved_bert_tokenizer")
    model_bert = BertForSequenceClassification.from_pretrained("./saved_bert_model")
    model_bert.eval()

    def predict_dialog_act(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        return predicted_class_id

    # This function is defined but never called, so this code doesn't run

try:
    while True:
        user_input = input("YOU: ")
        if user_input.lower() == "exit":
            print("BOT: Goodbye!")
            break

        # You could call bert_code_for_display_only() here if wanted,
        # but for now we skip running any BERT code.

        response = chat.send_message(user_input)
        print("BOT:", response.text)

except Exception as e:
    print(f"An error occurred: {e}")