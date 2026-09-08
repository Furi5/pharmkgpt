# PharmKGPT local release adaptation, 2026-09-08. See kg_construction/SOURCE_NOTES.md.
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Union, List
import numpy as np

class LangchainOutputParser:
    """
    A parser class for extracting and embedding information using Langchain and OpenAI APIs.
    """

    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5) -> None:
        """
        Initialize the LangchainOutputParser with specified API key, models, and operational parameters.
        
        Args:
        api_key (str): The API key for accessing OpenAI services.
        embeddings_model_name (str): The model name for text embeddings.
        model_name (str): The model name for the Chat API.
        temperature (float): The temperature setting for the Chat API's responses.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors.
        """
        self.model = llm_model

        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time

    def calculate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate embeddings for the given text using the initialized embeddings model.
        
        Args:
        text (Union[str, List[str]]): The text or list of texts to embed.
        
        Returns:
        np.ndarray: The calculated embeddings as a NumPy array.
        
        Raises:
        TypeError: If the input text is neither a string nor a list of strings.
        """
        if isinstance(text, list):
            return np.array(self.embeddings_model.embed_documents(text))
        elif isinstance(text, str):
            return np.array(self.embeddings_model.embed_query(text))
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")

    def extract_information_as_json_for_context(
        self,
        output_data_structure,
        context: str,
        IE_query: str = '''
        # DIRECTIVES : 
        - Act like an experienced information extractor. 
        - If you do not find the right information, keep its place empty.
        '''
        ):
        """
        Extract information from a given context and format it as JSON using a specified structure.
        
        Args:
        output_data_structure: The data structure definition for formatting the JSON output.
        context (str): The context from which to extract information.
        IE_query (str): The query to provide to the language model for extracting information.
        
        Returns:
        The structured JSON output based on the provided data structure and extracted information.
        
        Note: Handles rate limit and bad request errors by waiting and retrying.
        """
        # Set up a parser and inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=output_data_structure)

        template = f"""
        Context: {context}

        Question: {{query}}
        Format_instructions : {{format_instructions}}
        Answer: """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser
        # Extraction stages own the bounded retry loops. Preserve their arguments
        # and let errors propagate instead of recursively dropping the schema.
        return chain.invoke({"query": IE_query})
