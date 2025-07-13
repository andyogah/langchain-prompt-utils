"""
LangChain workflow integration examples using only official LangChain APIs.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import os

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator, EvaluatorType

DEFAULT_MODEL = os.environ.get("PROMPT_ENGINEERING_MODEL", "gpt-3.5-turbo")
DEFAULT_MODEL_ADVANCED = os.environ.get("PROMPT_ENGINEERING_MODEL_ADVANCED", "gpt-4")

class LangChainWorkflowIntegration:
    """
    LangChain workflow integration patterns using only official APIs.
    """

    def __init__(self, model: str = DEFAULT_MODEL, advanced_model: str = DEFAULT_MODEL_ADVANCED):
        self.model = model
        self.advanced_model = advanced_model
        self.llm = ChatOpenAI(model=self.model)
        self.llm_advanced = ChatOpenAI(model=self.advanced_model)
        self.logger = logging.getLogger(__name__)

    def langchain_rag_pipeline(self, query: str, documents: List[str]) -> Dict[str, Any]:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(documents, embeddings)
        prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_advanced,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )
            result = qa_chain.run(query)
        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}")
            result = None
        return {
            "query": query,
            "answer": result,
            "retrieval_method": "langchain_rag",
            "timestamp": datetime.now().isoformat()
        }

    def langchain_agent_workflow(self, task: str) -> Dict[str, Any]:
        def analyze_task(task_input: str) -> str:
            prompt = PromptTemplate(
                template="Analyze this task: {task}\nAnalysis:",
                input_variables=["task"]
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(task=task_input)
        def create_plan(analysis: str) -> str:
            prompt = PromptTemplate(
                template="Based on this analysis: {analysis}\nCreate a plan:",
                input_variables=["analysis"]
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(analysis=analysis)
        tools = [
            Tool(
                name="Task Analyzer",
                func=analyze_task,
                description="Analyze tasks and break them down"
            ),
            Tool(
                name="Plan Creator",
                func=create_plan,
                description="Create execution plans based on analysis"
            )
        ]
        try:
            agent = initialize_agent(
                tools=tools,
                llm=self.llm_advanced,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            result = agent.run(task)
        except Exception as e:
            self.logger.error(f"Agent workflow failed: {e}")
            result = None
        return {
            "task": task,
            "agent_result": result,
            "tools_used": [tool.name for tool in tools],
            "timestamp": datetime.now().isoformat()
        }

    def langchain_sequential_pipeline(self, input_text: str) -> Dict[str, Any]:
        prompt_entities = PromptTemplate(
            template="Extract key entities from this text: {text}\nEntities:",
            input_variables=["text"]
        )
        prompt_sentiment = PromptTemplate(
            template="Analyze sentiment of this text: {text}\nSentiment:",
            input_variables=["text"]
        )
        prompt_summary = PromptTemplate(
            template="""Based on the entities: {entities}
And sentiment: {sentiment}

Summarize the original text: {text}

Summary:""",
            input_variables=["entities", "sentiment", "text"]
        )
        try:
            entity_chain = LLMChain(
                llm=self.llm_advanced,
                prompt=prompt_entities,
                output_key="entities"
            )
            sentiment_chain = LLMChain(
                llm=self.llm_advanced,
                prompt=prompt_sentiment,
                output_key="sentiment"
            )
            summary_chain = LLMChain(
                llm=self.llm_advanced,
                prompt=prompt_summary,
                output_key="summary"
            )
            overall_chain = SequentialChain(
                chains=[entity_chain, sentiment_chain, summary_chain],
                input_variables=["text"],
                output_variables=["entities", "sentiment", "summary"],
                verbose=True
            )
            result = overall_chain({"text": input_text})
        except Exception as e:
            self.logger.error(f"Sequential pipeline failed: {e}")
            result = {"entities": None, "sentiment": None, "summary": None}
        return {
            "input_text": input_text,
            "entities": result.get("entities"),
            "sentiment": result.get("sentiment"),
            "summary": result.get("summary"),
            "timestamp": datetime.now().isoformat()
        }

    def langchain_conversation_memory(self, user_input: str, conversation_id: str) -> Dict[str, Any]:
        memory = ConversationBufferMemory(return_messages=True)
        prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Have a natural conversation.

Current conversation:
{history}

Human: {input}
AI:""",
            input_variables=["history", "input"]
        )
        try:
            conversation_chain = ConversationChain(
                llm=self.llm,
                memory=memory,
                prompt=prompt,
                verbose=True
            )
            response = conversation_chain.predict(input=user_input)
        except Exception as e:
            self.logger.error(f"Conversation memory failed: {e}")
            response = None
        return {
            "conversation_id": conversation_id,
            "user_input": user_input,
            "ai_response": response,
            "memory_length": len(memory.chat_memory.messages),
            "timestamp": datetime.now().isoformat()
        }

    def langchain_evaluation(self, input_text: str, prediction: str, reference: str = None) -> Dict[str, Any]:
        evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=self.llm, criteria="helpfulness")
        try:
            result = evaluator.evaluate_strings(
                input=input_text,
                prediction=prediction,
                reference=reference
            )
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            result = {"score": 0.0, "reasoning": str(e)}
        return result

def main():
    logging.basicConfig(level=logging.INFO)
    workflow = LangChainWorkflowIntegration()
    print("=== LangChain RAG Pipeline ===")
    rag_result = workflow.langchain_rag_pipeline(
        query="What are the benefits of renewable energy?",
        documents=[
            "Renewable energy reduces carbon emissions and helps fight climate change.",
            "Solar and wind power are becoming increasingly cost-effective.",
            "Renewable energy creates jobs in manufacturing and installation."
        ]
    )
    print(f"RAG Answer: {rag_result['answer'][:100]}...")

    print("\n=== LangChain Agent Workflow ===")
    agent_result = workflow.langchain_agent_workflow(
        task="Create a marketing strategy for a new eco-friendly product"
    )
    print(f"Agent Result: {agent_result['agent_result'][:100]}...")

    print("\n=== LangChain Sequential Pipeline ===")
    pipeline_result = workflow.langchain_sequential_pipeline(
        input_text="I love this new product! It's amazing and works perfectly."
    )
    print(f"Entities: {pipeline_result['entities'][:50]}...")
    print(f"Sentiment: {pipeline_result['sentiment'][:50]}...")
    print(f"Summary: {pipeline_result['summary'][:100]}...")

    print("\n=== LangChain Conversation with Memory ===")
    conversation_result = workflow.langchain_conversation_memory(
        user_input="Hello, I'm interested in learning about AI.",
        conversation_id="conv_001"
    )
    print(f"AI Response: {conversation_result['ai_response'][:100]}...")

    print("\n=== LangChain Workflow Integration Examples Complete ===")

if __name__ == "__main__":
    main()
