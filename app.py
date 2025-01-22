import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import google.generativeai as genai
import snowflake.connector
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware



# Load environment variables
load_dotenv()

# FastAPI initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize template engine
templates = Jinja2Templates(directory="templates")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Snowflake configuration
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# User input schema
class PromptRequest(BaseModel):
    prompt: str
    table: str

# Function to clean SQL query
def clean_sql_query(query: str) -> str:
    query = query.replace("`", "")
    query = re.sub(r'\bSQL\b', '', query, flags=re.IGNORECASE).strip()
    return query

# Function to generate SQL query using Gemini Pro
def generate_sql_query(prompt: str):
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return clean_sql_query(response.text.strip())
        else:
            raise ValueError("Failed to generate SQL query.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Pro error: {e}")

# Function to query Snowflake
def execute_query(sql_query: str):
    conn = None
    try:
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snowflake error: {e}")
    finally:
        if conn:
            conn.close()

# Function to format query results using LLM
def format_results_with_llm( prompt: str,query_results):
    try:
        result_text = f"Query Results: {query_results}"
        llm_prompt = (
            f"Based on the query '{prompt}', the results are:\n{result_text}\n"
            "Please summarize this information in a single plain-text sentence without special characters, formatting,and line breaks."
        )   
        
        response = model.generate_content(llm_prompt)
        plain_text_result = response.text.strip() if response and response.text else "Failed to format results."
        return plain_text_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Pro formatting error: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/prompt-to-query")
async def prompt_to_query(request: PromptRequest):
    try:
        # Validate input
        if not request.table or not request.prompt:
            raise HTTPException(status_code=400, detail="Table name and prompt are required.")
        
        # Include the selected table in the prompt
        full_prompt = f"For the table '{request.table}', {request.prompt}"

        # Step 1: Generate SQL query
        sql_query = generate_sql_query(full_prompt)
        print(f"Generated SQL Query: {sql_query}")

        # Step 2: Execute the query
        query_results = execute_query(sql_query)
        print(f"Query Results: {query_results}")

        # Step 3: Use LLM to format the query results
        formatted_response = format_results_with_llm(request, query_results)

        # Step 4: Return the response
        return {
            "prompt": request.prompt,
            "sql_query": sql_query,
            "formatted_response": formatted_response,
        }
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
