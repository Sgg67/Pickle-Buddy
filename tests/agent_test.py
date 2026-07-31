# import necessary modules
from dotenv import load_dotenv
from backend.core import run_llm

# test if the source of the question is valid
def test_valid_question_source() -> None:
    # have a question that you will prompt the llm with
    question = "Who is anna leah waters"
    # call the llm with the question
    response = run_llm(question)
    
    # Extract sources from metadata
    sources = [doc.metadata.get("source") for doc in response["context"]]
    
    # Check if the URL is in the retrieved sources
    assert "https://www.11pickles.com/post/best-pickleball-players" in sources

# test to show there are no sources for an off topic question
def test_off_topic_question_source() -> None:
    # prompt of question
    question = "best pizza in pittsburgh" 
    # ask llm question
    response = run_llm(question)
    # have llm tell you they can not answer the question
    assert "cannot" in response["answer"].lower() or "pickleball" in response["answer"].lower()

# test some information in llm response
def test_valid_question_answer() -> None:
    # form the question
    question = "Who is anna leah waters"
    # ask the llm the question
    response = run_llm(question)
    # store the response in lowercase
    answer = response["answer"].lower()
    # assert information that should be in llm's response
    assert "skechers" in answer
    assert "ppa" in answer