from retriever import Retriever
from langgraph.graph import START, StateGraph
from models import State

def create_chain():
    retriever = Retriever()

    """Create a LangGraph chain with validation."""
    # Create StateGraph with sequential execution
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("analyze", retriever.analyze_query)
    graph_builder.add_node("retrieve", retriever.retrieve)
    graph_builder.add_node("generate", retriever.generate)
    graph_builder.add_node("reject", retriever.reject_query)
    
    # Add edges with a clear sequential flow
    graph_builder.add_edge(START, "analyze")
    
    # Continue the sequential flow
    graph_builder.add_edge("analyze", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Set reject and generate as finish points
    graph_builder.set_finish_point("reject")
    graph_builder.set_finish_point("generate")
    
    return graph_builder.compile()

chain = create_chain()

async def documents_tool(state: State):
    """Tool for fetching documents from the knowledge base."""
    result = await chain.ainvoke(state)

    return result