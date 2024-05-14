from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAI
import os
from typing import Dict

class NLPManager:
    def __init__(self):
        # Initialize the OpenAI model
        self.model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

        # Define your desired data structure.
        class JSONObj(BaseModel):
            target: str = Field(description="identified target")
            heading: str = Field(description="convert the heading number to integers instead of numbers in words")
            tool: str = Field(description="tool used to attack target")

        # Set up a parser + inject instructions into the prompt template.
        self.parser = PydanticOutputParser(pydantic_object=JSONObj)

        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def qa(self, context: str):
        # Run prompt and model
        prompt_and_model = self.prompt | self.model
        output = prompt_and_model.invoke({"query": context})
        result = self.parser.invoke(output)
        return {"heading": result.heading, "tool": result.tool, "target": result.target}

if __name__ == "__main__":
    nlp_manager = NLPManager()
    inputs = ["We have a blue drone coming in from a heading of zero four five. Let's get those surface-to-air missiles ready.", "There's a black helicopter on a heading of 355. Ready the laser-guided missiles."]
    for context in inputs:
        result = nlp_manager.qa(context)
        print(result)
