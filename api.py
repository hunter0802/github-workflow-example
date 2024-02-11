import os
import pinecone
import json
import time
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
from openai import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from app.utils.func import chunks, setup_logger
from ..config import DATA_SRC_DIR
import mysql.connector

logger = setup_logger('app.log')

api = Blueprint('api', __name__)
a;
load_dotenv()

# initialize connection
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=os.getenv("DB_PORT")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# # initialize pinecone
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENV"),
# )

index_name = "chatbot"

pinecone_index = pc.Index(index_name)

# # Create embeddings
# embeddings = OpenAIEmbeddings()

# # API Endpoint to train the model
@api.route('/api/v1/train', methods=['POST'])
def train():
    # Ensure that method is POST
    if request.method == 'POST':
        # Check if gpt_key and site_url are provided
        if 'gpt_key' not in request.form or not request.form['gpt_key']:
            return jsonify({'message': 'Bad Request: Missing OpenAI Secret in the request'}), 400

        # Check if gpt_key and site_url are provided
        if 'user_id' not in request.form or not request.form['user_id']:
            return jsonify({'message': 'Bad Request: could not generate training data.'}), 400
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400

        file = request.files['file']
        # If no file is selected
        if file.filename == '':
            return jsonify({'message': 'No file selected for uploading'}), 400
        
        # Save the file
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(DATA_SRC_DIR, filename))

            # File saved. Now loading the file for training.
            file_path = os.path.join(DATA_SRC_DIR, filename) # Maybe we can optimize this line
            loader = TextLoader(file_path)
            documents = loader.load()

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            # Create embeddings
            os.environ["OPENAI_API_KEY"] = request.form['gpt_key']

            embeddings = OpenAIEmbeddings()

            print("before embedding")

            # Create a map object that generates tuples of (id, vector, metadata)
            doc_vectors = map(lambda idx, doc: (
                f'id-{idx}',  # ID
                embeddings.embed_documents([doc.page_content])[0],  # Vector
                {"text": doc.page_content}  # Metadata (replace with actual metadata)
            ), range(len(docs)), docs)

            # First, check if our index already exists. If it doesn't, we create it
            # if index_name not in pc.list_indexes():
            #     # we create a new index
            #     pc.create_index(
            #         name=index_name,
            #         metric='cosine',
            #         dimension=1536
            #     )

            # wait for index to be initialized
            # while not pc.describe_index(index_name).status['ready']:
            #     time.sleep(1)

            # Remove file extension
            base_filename = filename.rsplit('.', 1)[0]

            # Format the namespace by appending user_id
            namespace = f"{base_filename}_{request.form['user_id']}"

            print(f"namespace: {namespace}")

            # Upsert data with 100 vectors per upsert request
            # for doc_vector in chunks(doc_vectors, batch_size=100):
            pinecone_index.upsert(vectors=doc_vectors, namespace=namespace, show_progress=True)

            return jsonify({'message': 'Vectorstore generated successfully' }), 200
    else:
        return jsonify({'message': 'Bad Request: This endpoint only accepts POST requests'}), 400

# response v2
@api.route('/api/v2/response', methods=['GET', 'POST'])
def response_v2():
    if request.method == 'POST':
        try:
            data = request.get_json()

            if 'query' not in data and 'gpt_key' not in data:
                return jsonify({'message': 'Bad Request: Missing query or OpenAI Secret in request'}), 400

            if 'namespace' not in data:
                return jsonify({'message': 'Bad Request: It seems you have not trained any bot with your source yet.'}), 400

            os.environ["OPENAI_API_KEY"] = data['gpt_key']

            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            vectors = pinecone_index.query(
                namespace=data['namespace'],
                include_metadata=True,
                top_k=3,
                vector=embeddings.embed_query(data['query'])
            )
            
            retrived_context = ""
            for vector in vectors['matches']:
                retrived_context = retrived_context + vector['metadata']['text'] + "\n\n"

            # chat_history = data['chat_history']
            # namespace = data['namespace']
            client = OpenAI(api_key=data['gpt_key'])
            messages = [
                {
                    "role": "system",
                    "content": f"""
You are a sales assistant. You have access to WooCommerce database and your role is to answer questions of the customers. Database is typical WooCommerce Wordpress Database.
Also, you can refer to the context when you answer.

This is the context you can refer.
-------
{retrived_context}
"""
                },
                {
                    "role": "user",
                    "content": data['query']
                }
            ]
            chat_completion = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=[{
                    "name": "search_database",
                    "description": "Search through typical WooCommerce Wordpress database. Database schema is typical WooCommerce Wordpress database schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_list": {
                                "type": "boolean",
                                "description": "Indicate if answer to the user question is list or specific one value. If list, it's true. Otherwise: false"
                            }
                        },
                        "required": ["is_list"]
                    }
                }],
                temperature=0
            )
            response_message = chat_completion.choices[0].message
            function_call = response_message.function_call
            if function_call:
                argument = json.loads(function_call.arguments)
                print(argument)
                chat_completion_sql_query = client.chat.completions.create(
                    model="gpt-4",
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
    {data['query']} \n 

    Generate SQL query for this question. 
    Database schema is typical WooCommerce Wordpress schema without any customization. 
    Follow this step to generate accurate SQL query.
    1. Determine which tables you need to answer the question
    2. Check schemas of tables you need.
    3. Determine which field user would want to see.
    4. Write an SQL query.

    Note: SQL query should have all static values, not any variables. SQL query must be executable directly without any replacement.
    """
                        }
                    ]
                )

                print(chat_completion_sql_query.choices[0].message.content)
                # extract only sql
                extract_completion_sql_query = client.chat.completions.create(
                    model="gpt-4",
                    temperature=0,
                    messages=[{"role": "user", "content": chat_completion_sql_query.choices[0].message.content}],
                    functions=[{
                        "name": "extract_sql_query",
                        "description": "Extract sql query from the user message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sql": {
                                    "type": "string",
                                    "description": "Extracted sql query in correct sql format. It shouldn't include '\n'"
                                }
                            },
                            "required": ["sql"]
                        }
                    }],
                    function_call={"name": "extract_sql_query"}
                )

                sql_extract_function_call = extract_completion_sql_query.choices[0].message.function_call

                if sql_extract_function_call:
                    sql_argument = json.loads(sql_extract_function_call.arguments)
                    
                    # execute sql query
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(sql_argument["sql"])
                    results = cursor.fetchall()
                    cursor.close()
                    # results = [{
                    #     "total_stock": "25.0"
                    # }]

                    if argument["is_list"] == True:
                        return jsonify({
                            "data": results,
                            "type": "list"
                        })
                    else:
                        # make natural language answer
                        natural_response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"{data['gpt_key']} \n This is Database Query Result: {json.dumps(results)} \n Don't mention that you're answering from the query result."
                                }
                            ]
                        )
                        return jsonify({
                            "data": natural_response.choices[0].message.content,
                            "type": "text"
                        })
                else:
                    return jsonify({'message': 'Can not extract sql from'}), 500
        except Exception as e:
            print(e)
            return jsonify({
                "data": "Sorry, but something's went wrong. Please try again!",
                "type": "text"
            })
        else:
            return jsonify({
                "data": response_message.content,
                "type": "text"
            })

    else:
        return jsonify({'message': 'Bad Request: This endpoint only accepts POST requests'}), 400

# # API Endpoint to interact with request
# @api.route('/api/v1/response', methods=['GET', 'POST'])
# def response_v1():

#     if request.method == 'POST':
#         data = request.get_json()

#         if 'query' not in data and 'gpt_key' not in data:
#             return jsonify({'message': 'Bad Request: Missing query or OpenAI Secret in request'}), 400

#         if 'namespace' not in data:
#             return jsonify({'message': 'Bad Request: It seems you have not trained any bot with your source yet.'}), 400

#         os.environ["OPENAI_API_KEY"] = data['gpt_key']

#         # Get query
#         query = data['query']
#         chat_history = data['chat_history']
#         namespace = data['namespace']

#         # Create embeddings
#         embeddings = OpenAIEmbeddings()

#         vectorstore = Pinecone(
#             pinecone_index, embeddings.embed_query, "text"
#         )

#         similarity_search = vectorstore.similarity_search(
#             query,  # our search query
#             k=3,  # return 3 most relevant docs
#             namespace=namespace
#         )

#         # Convert dictionary to a JSON-formatted string.
#         # Could be improved later
#         chat_history_json_string = json.dumps(chat_history)

#         template = """You are an intelligent chatbot having a conversation with a human.

#         Given the following extracted parts of a long document and a question, create a final answer.

#         You also answer from the previous chat history provided below. You're answer should be like human reply.

#         If anything asked out of the context provided below, simply reply that the question isn't in the context you are trained for.

#         {context}

#         Chat history below:

#         {chat_history}

#         Human: {human_input}
#         Chatbot:"""

#         prompt = PromptTemplate(
#             input_variables=["chat_history", "human_input", "context"], template=template
#         )

#         # We're passing the previous chat history in the prompt instead of using memory
#         # memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
#         chain = load_qa_chain(
#             OpenAI(temperature=0), chain_type="stuff", prompt=prompt
#         )

#         # Track the token usages and costs
#         with get_openai_callback() as cb:

#             answer = chain({"input_documents": similarity_search, "human_input": query, "chat_history": chat_history_json_string}, return_only_outputs=True)

#             # Create a dictionary with the callback responses
#             cb_responses = {
#                 "total_tokens": cb.total_tokens,
#                 "prompt_tokens": cb.prompt_tokens,
#                 "completion_tokens": cb.completion_tokens,
#                 "successful_requests": cb.successful_requests,
#                 "total_cost": cb.total_cost,
#             }

#         data = {
#             "message": answer,
#             "token_usage": cb_responses
#         }

#         return jsonify(data)

#     else:
#         return jsonify({'message': 'Bad Request: This endpoint only accepts POST requests'}), 400