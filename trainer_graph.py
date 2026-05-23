import os
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Database Configuration
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")
PG_DB = os.getenv("PG_DB", "health")

def get_db_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB,
        cursor_factory=RealDictCursor
    )

class GraphState(TypedDict):
    target_pace: float
    current_avg_pace: float
    drift_status: str
    suggested_adjustments: str
    week_number: int

# --- Nodes ---

def fetch_performance_node(state: GraphState):
    """Fetch the average pace of Running (ID 37) from the last week."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Joining workouts and splits to get pace for the most recent week
    query = """
        SELECT AVG(s.avg_pace_seconds_per_km) as avg_pace
        FROM workout_splits s
        JOIN workouts w ON s.workout_id = w.id
        WHERE w.activity_type = 37 
        AND w.start_date > CURRENT_DATE - INTERVAL '7 days'
    """
    cur.execute(query)
    res = cur.fetchone()
    conn.close()
    
    # If no runs found, we use your baseline pace (~490s/km from ID 1)
    pace = res['avg_pace'] if res['avg_pace'] else 490.0
    return {"current_avg_pace": round(float(pace), 2)}

def analyze_drift_node(state: GraphState):
    """Gemini-powered analysis of the gap between current pace and 60m target."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    target = 360.0  # 6:00 min/km
    current = state["current_avg_pace"]
    diff = current - target
    
    prompt = f"""
    You are a professional running coach.
    Target Pace: {target} s/km (6:00 min/km).
    Current Avg Pace: {current} s/km.
    Gap: {diff} seconds.
    
    Based on this gap, provide a short 'Drift Analysis'. 
    If they are within 10s of target, praise them. 
    If they are >60s slower, suggest focusing on Zone 2 heart rate runs and Functional Strength (ID 20).
    Be concise (max 3 sentences).
    """
    
    response = llm.invoke(prompt)
    return {
        "drift_status": "Losing Ground" if diff > 10 else "Gaining Momentum", 
        "suggested_adjustments": response.content
    }

def update_plan_node(state: GraphState):
    """Persist the adjustment and drift status back into the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Ensure you have the training_plans table created first!
    query = """
        INSERT INTO training_plans (goal_name, current_status, last_updated)
        VALUES ('10k_60mins', %s, NOW())
        ON CONFLICT (goal_name) 
        DO UPDATE SET current_status = EXCLUDED.current_status, last_updated = NOW();
    """
    cur.execute(query, (state["suggested_adjustments"],))
    conn.commit()
    conn.close()
    
    print(f"--- Weekly Check Complete ---")
    print(f"Status: {state['drift_status']} | Current Pace: {state['current_avg_pace']}s/km")
    return state

# --- Graph Assembly ---

workflow = StateGraph(GraphState)

workflow.add_node("fetch_performance", fetch_performance_node)
workflow.add_node("analyze_drift", analyze_drift_node)
workflow.add_node("update_plan", update_plan_node)

workflow.set_entry_point("fetch_performance")
workflow.add_edge("fetch_performance", "analyze_drift")
workflow.add_edge("analyze_drift", "update_plan")
workflow.add_edge("update_plan", END)

app = workflow.compile()