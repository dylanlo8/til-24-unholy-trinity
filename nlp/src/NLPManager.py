from typing import Dict
from transformers import DistilBertForQuestionAnswering, AutoTokenizer
import torch
from fuzzywuzzy import process
import os

class NLPManager:
    def __init__(self):
        # initialize the model here
        model_path = "distilbert-base-cased-distilled-chaii"
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # self.model = pipeline("question-answering", "SauravMaheshkar/distilbert-base-cased-distilled-chaii")
        pass

    def qa(self, context: str):
        # heading = self.model(question="What is the heading?", context=context)
        # heading= self.convert_words_to_number(heading['answer'])
        # target = self.model(question="What is the target?", context=context)
        # target= target['answer']
        # tool = self.model(question="What is the tool to deploy?", context=context)
        # tool= tool['answer']
        # perform NLP question-answering
        # Tokenize the context to find the exact start and end position of the answer
        inputs = self.tokenizer.encode_plus("What is the heading?", context, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        heading_start = torch.argmax(outputs.start_logits, dim=1).item()
        heading_end = torch.argmax(outputs.end_logits, dim=1).item() + 1
        heading = self.tokenizer.decode(inputs["input_ids"][0][heading_start:heading_end])
        heading = self.convert_words_to_number(heading)
        
        inputs = self.tokenizer.encode_plus("What is the tool?", context, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        tool_start = torch.argmax(outputs.start_logits, dim=1).item()
        tool_end = torch.argmax(outputs.end_logits, dim=1).item() + 1
        tool = self.tokenizer.decode(inputs["input_ids"][0][tool_start:tool_end])
        tool = self.remove_whitespace(tool)
        
        inputs = self.tokenizer.encode_plus("What is the target?", context, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_start = torch.argmax(outputs.start_logits, dim=1).item()
        target_end = torch.argmax(outputs.end_logits, dim=1).item() + 1
        target = self.tokenizer.decode(inputs["input_ids"][0][target_start:target_end])
        target = self.remove_whitespace(target)

        return {"heading": heading, "tool": tool, "target": target}
        # return {"heading": '', "tool": '', "target": ''}

    def convert_words_to_number(self, words):
        word_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        number = ''
        for word in words.split():
            closest_match = process.extractOne(word, word_mapping.keys())[0]
            number += word_mapping[closest_match]
        return number

    def remove_whitespace(self, words):
        # Remove whitespace around hyphens
        words = words.replace(" - ", "-")
        return words
    
if __name__ == "__main__":
    nlp_manager = NLPManager()
    inputs = "tool to deploy is big, black nuclear bombs, target is yellow, pink pokka-dotted fisher fighter jet, heading is sevenin niner throne,."
    result = nlp_manager.qa(inputs)
    print(result)
