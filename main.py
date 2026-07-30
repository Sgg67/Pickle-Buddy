# import necessary dependencies
from typing import Any, Dict, List
import uuid
import streamlit as st
from backend.core import run_llm

# format the sources so they look presentable
def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        # convert the source into string format if it exists
        str((meta.get("source") or "Unknown"))
        # loop through context docs if it exists
        for doc in (context_docs or [])
        # if the attribute exists assign metadata to the meta variable
        if(meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="Pickle Buddy", layout="centered")
st.title("Pickle Buddy")

# add a sidebar that clears the chat and session
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# if there are no messages the session
if "messages" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # have the ai send a message prompting the user to ask the ai about pickleball
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about pickleball",
            "sources": [
                    "https://pickleballkitchen.com/14-effective-shots-use-pickleball/",
                    "https://happypeoplepickleball.com/best-places-to-play-pickleball/",
                    "https://www.11pickles.com/post/best-pickleball-players",
                    "https://thekitchenpickle.com/blogs/gear/best-pickleball-paddles-available/",
                    "USAP-Official-Rulebook"
            ],
        }
    ]

# loop through the messages in the session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # add the content to the screen
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about Pickleball..")

# if there is a prompt
if prompt:
    # append the user and the content of teh prompt
    st.session_state.messages.append({"role":"user", "content": prompt, "sources": []})
    # write the chat message that the user writes to the screen
    with st.chat_message("user"):
        st.markdown(prompt)

    # if it is the ai's turn to chat
    with st.chat_message("assistant"):
        # try to generate a response for the user
        try:
            # while the ai is retrieving the answer have a spinner on the screen and say the chat bot is generating the answer
            with st.spinner("Retrieving docs about pickleball and generating answer"):
                # get the result as a dictionary
                result: Dict[str, Any] = run_llm(prompt, thread_id=st.session_state.get("session_id", f"session"))
                # get the answer if it exists otherwise return that no answer exists in the documents
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                # format the sources
                sources = _format_sources(result.get("context", []))


                # return the answer that the ai gives
                st.markdown(answer)

                # store all the session messages giving the ai context
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
        # otherwise return an error
        except Exception as e:
            st.error("Failed to generate a response, try again later.")