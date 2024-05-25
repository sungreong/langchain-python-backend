from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import sqlite3

# Read csv
import pandas as pd
import re

if __name__ == "__main__":

    llm = ChatOpenAI(
        model_name="microsoft/Phi-3-mini-4k-instruct",  # Ensure this matches a model your server can serve
        openai_api_base="http://localhost:8000/v1",  # Change this to your local server's URL
        openai_api_key="Not needed for local server",  # API key not required for local setups
        openai_proxy="",  # Endpoint for generating responses
        temperature=0.0,  # Deterministic output,
        max_tokens=3000,  # Maximum tokens to generate
        stream=False,
        # request_timeout=120,  # Timeout for requests
        frequency_penalty=0.0,
        # presence_penalty=0.8,
        top_p=1.0,
        # best_of=1,
        # n=1,
        # # 차이가 없음
        # logit_bias={},
        # logprobs=0,
        # # 차이가 없음
        # seed = 0,
        # stop=[],
        # echo=True,
        # initial_agent
    )

    # Connect to database
    conn = sqlite3.connect("./../../data/pincode.db")
    cursor = conn.cursor()

    # Create table
    query = """
    CREATE TABLE Postal_Offices (
        CircleName VARCHAR(255),
        RegionName VARCHAR(255),
        DivisionName VARCHAR(255),
        OfficeName VARCHAR(255),
        Pincode INTEGER,
        OfficeType VARCHAR(255),
        Delivery VARCHAR(255),
        District VARCHAR(255),
        StateName VARCHAR(255)
    );

    """
    try:
        cursor.execute(query)
    except Exception as e:
        print(e)

    df = pd.read_csv("./../../data/Pincode_30052019.csv", encoding="ISO-8859-1")
    df.columns = [
        "CircleName",
        "RegionName",
        "DivisionName",
        "OfficeName",
        "Pincode",
        "OfficeType",
        "Delivery",
        "District",
        "StateName",
    ]

    # Import the csv into database
    try:
        df.to_sql("Postal_Offices", conn, if_exists="fail", index=False)
    except Exception as e:
        print(e)
    finally:
        conn.close()

    db = SQLDatabase.from_uri("sqlite:///../../data/pincode.db")
    chain = create_sql_query_chain(llm, db)
    response = chain.invoke({"question": "What is address of pincode 800020"})
    print(response)
    # Use regular expression to extract the SQL query
    sql_query = re.search(r"SQLQuery: (.+)SQLResult", response).group(1).strip()
    print(sql_query)
    cursor.execute(sql_query)
    print(cursor.fetchall())
