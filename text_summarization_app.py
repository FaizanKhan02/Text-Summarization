from tkinter import *
from tkinter import filedialog
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# Initialize main application window
main = Tk()
main.title("Abstractive Text Summarization")
main.geometry("1000x700")
main.config(bg='Darkcyan')

# Global variables
model, tokenizer, device = None, None, None

# Initialize tokenizer
tokenizer = None  # Initialize tokenizer variable

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    tokens = re.sub(r'\d+', '', tokens)
    return tokens

def loadModel():
    global model, tokenizer, device
    try:
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')
        pathlabel.config(text="Transformer Model Loaded Successfully")
        output_text.delete('1.0', END)
        output_text.insert(END, f"Model loaded successfully.\n")
    except Exception as e:
        pathlabel.config(text="Error loading model")
        output_text.insert(END, f"Error loading model: {str(e)}\n")

def preprocessText():
    text_data = input_text.get("1.0", "end-1c").strip()
    if not text_data:
        output_text.delete('1.0', END)
        output_text.insert(END, "Error: No text to preprocess.\n")
        return
        cleaned_text = cleanText(text_data)
        output_text.delete('1.0', END)
        output_text.insert(END, cleaned_text)
    
def abstractiveSummary():
    if model is None or tokenizer is None:
        output_text.delete('1.0', END)
        output_text.insert(END, "Error: Load the model first.\n")
        return

    text_data = input_text.get("1.0", "end-1c").strip()
    if not text_data:
        output_text.delete('1.0', END)
        output_text.insert(END, "Error: No text provided for summarization.\n")
        return

    try:
        input_text_processed = "summarize: " + text_data
        tokenized_text = tokenizer.encode(input_text_processed, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(tokenized_text, max_length=120, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        output_text.delete('1.0', END)
        output_text.insert(END, summary)
    except Exception as e:
        output_text.delete('1.0', END)
        output_text.insert(END, f"Error generating summary: {str(e)}\n")


def clearText():
    input_text.delete('1.0', END)
    output_text.delete('1.0', END)

# Setting up the layout
font_title = ('times', 20, 'bold')
font_label = ('times', 13, 'bold')
font_text = ('times', 12, 'normal')

# Title
title = Label(main, text='Abstractive Text Summarization', bg='lightcyan', fg='Black', font=font_title, height=2, width=150)
title.pack()

# Load Model Button
load_button = Button(main, text="Load Transformer Model", command=loadModel, font=font_label, cursor='hand2')
load_button.pack(pady=10)

# Path label for model status
pathlabel = Label(main, bg='Darkcyan', fg='white', font=font_label)
pathlabel.pack()

# Input Text Area
input_label = Label(main, text='Input Your Text Below:', bg='Darkcyan', fg='white', font=font_label)
input_label.pack(pady=5)

input_text = Text(main, height=13, width=120, font=font_text)
input_text.pack(pady=10)

# Buttons for preprocessing, summarization, and clearing text
button_frame = Frame(main, bg='Darkcyan')
button_frame.pack(pady=10)

preprocess_button = Button(button_frame, text="Preprocess Text", command=preprocessText, font=font_label, cursor='hand2')
preprocess_button.grid(row=0, column=0, padx=10)

summary_button = Button(button_frame, text="Generate Summary", command=abstractiveSummary, font=font_label, cursor='hand2')
summary_button.grid(row=0, column=1, padx=10)

clear_button = Button(button_frame, text="Clear Text", command=clearText, font=font_label, cursor='hand2')
clear_button.grid(row=0, column=2, padx=10)

# Output Text Area
output_label = Label(main, text='Output Summary:', bg='Darkcyan', fg='white', font=font_label)
output_label.pack(pady=5)

output_text = Text(main, height=13, width=120, font=font_text)
output_text.pack(pady=10)

# Main loop
main.mainloop()
