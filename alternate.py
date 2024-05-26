import os
import dotenv
import uuid
from openai import OpenAI
import streamlit as st
from timeit import default_timer as timer
from streamlit_chat import message
from neo4j import GraphDatabase
from pymongo import MongoClient
from datetime import datetime, timezone

# Load environment variables
dotenv.load_dotenv()

# Neo4j configuration
neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

# MongoDB configuration
mongo_url = os.getenv("MONGO_URL")

try:
    mongo_client = MongoClient(mongo_url)
    db = mongo_client["langchain"]
    session_collection = db["sessions"]
    print("MongoDB connection successful.")
except Exception as e:
    print(f"MongoDB connection failed: {e}")

# OpenAI configuration
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Neo4j driver
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

try:
    with driver.session() as session:
        result = session.run("RETURN 1")
        for record in result:
            print(record)
    print("Neo4j connection successful.")
except Exception as e:
    print(f"Neo4j connection failed: {e}")

# Cypher generation prompt
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Client, use `toLower(client.id) contains 'neo4j'`. To search for Slack Messages, use 'toLower(SlackMessage.text) contains 'neo4j'`. To search for a project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.
5. Never use relationships that are not mentioned in the given schema
6. When asked about projects, Match the properties using case-insensitive matching and the OR-operator, E.g, to find a logistics platform -project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.

schema: {schema}

Examples:
Question: Which client's projects use most of our people?
Answer: ```MATCH (c:Client)<-[:HAS_CLIENT]-(p:Project)-[:HAS_PEOPLE]->(person:Person)
RETURN c.name AS Client, COUNT(DISTINCT person) AS NumberOfPeople
ORDER BY NumberOfPeople DESC```
Question: Which person uses the largest number of different technologies?
Answer: ```MATCH (person:Person)-[:USES_TECH]->(tech:Technology)
RETURN person.name AS PersonName, COUNT(DISTINCT tech) AS NumberOfTechnologies
ORDER BY NumberOfTechnologies DESC```

Question: {question}

Previous Queries:
{previous_queries}
"""

# Function to get the schema from Neo4j
def get_schema():
    query = """
    CALL apoc.meta.schema()
    YIELD value
    RETURN value
    """
    with driver.session() as session:
        result = session.run(query)
        schema = result.single()["value"]
    return schema

# Function to generate Cypher query using OpenAI
def generate_cypher(schema, question, previous_queries):
    prompt = cypher_generation_template.format(schema=schema, question=question, previous_queries=previous_queries)
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert Neo4j Cypher translator."},
            {"role": "user", "content": prompt}
        ]
    )
    
    cypher_query = response.choices[0].message.content.strip()
    # Strip markdown formatting if present
    if cypher_query.startswith("```") and cypher_query.endswith("```"):
        cypher_query = cypher_query[3:-3].strip()

    print(f"Generated Cypher:\n{cypher_query}")
    return cypher_query

# Function to query Neo4j
def query_neo4j(cypher_query):
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        return None

# Function to log the process
def log_process(session_id, cypher_query):
    session = session_collection.find_one({"session_id": session_id})
    if session:
        session_collection.update_one({"session_id": session_id}, {"$push": {"queries": cypher_query}})
    else:
        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "session_id": session_id,
            "queries": [cypher_query]
        }
        session_collection.insert_one(log_entry)

# Custom Chain function with retry logic
def custom_chain(question, schema, session_id, retries=5):
    print("> Entering customChain...")
    session = session_collection.find_one({"session_id": session_id})
    previous_queries = "\n".join(session["queries"]) if session else "No previous queries."
    
    for attempt in range(retries):
        cypher_query = generate_cypher(schema, question, previous_queries)
        print(f"Generated Cypher (Attempt {attempt+1}):\n{cypher_query}")
        database_results = query_neo4j(cypher_query)
        if database_results is not None:
            print(f"Full Context:\n{database_results}")
            log_process(session_id, cypher_query)
            print("> Finished customChain.")
            return cypher_query, database_results
        print("Query failed, retrying...")
    raise Exception("Failed to generate a valid Cypher query after multiple attempts.")

# Function to format the database results
def format_results(database_results):
    if not database_results:
        return "No results found."
    
    formatted_result = ""
    for record in database_results:
        for key, value in record.items():
            formatted_result += f"{key}: {value}\n"
        formatted_result += "\n"
    return formatted_result

# Streamlit UI
st.set_page_config(layout="wide")

# Generate a new session ID if not already present
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

session_id = st.session_state.session_id

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("Conversational Neo4J Assistant")
    st.subheader(f"Session ID: {session_id}")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

user_input = st.text_input("Enter your question", key="input")
if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            # Get the schema from Neo4j
            schema = get_schema()
            cypher_query, database_results = custom_chain(user_input, schema, session_id)

            formatted_results = format_results(database_results)
            answer = f"Based on the provided information, here is the result:\n{formatted_results}"
            st.session_state.system_msgs.append(answer)

        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)
        finally:
            st.write(f"Time taken: {timer() - start:.2f}s")

        col1, col2 = st.columns([3, 1])

        # Display the chat history
        with col1:
            if st.session_state["system_msgs"]:
                for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                    message(st.session_state["system_msgs"][i], key=str(i) + "_assistant")
                    message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

        with col2:
            if 'cypher_query' in locals():
                st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
            if 'database_results' in locals():
                st.text_area("Last Database Results", formatted_results, key="_database", height=240)
