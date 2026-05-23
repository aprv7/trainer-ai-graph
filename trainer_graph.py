import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration - Ensure these are in your .env file
# GOOGLE_API_KEY=your_key_here
# DATABASE_URL=your_db_connection_string

class GraphState(TypedDict):
    target_pace: float
    current_avg_pace: float
    drift_status: str
    suggested_adjustments: str
    week_number: int

# --- Nodes ---

def fetch_performance_node(state: GraphState):
    """Queries the last 7 days of running data."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    # Calculate avg pace for Running (ID 37) in the last week
    cur.execute("""
        SELECT AVG(avg_pace_seconds_per_km) as avg_pace
        FROM workout_splits s
        JOIN workouts w ON s.workout_id = w.id
        WHERE w.activity_type = 37 
        AND w.start_date > CURRENT_DATE - INTERVAL '7 days'
    """)
    res = cur.fetchone()
    conn.close()
    
    pace = res['avg_pace'] if res['avg_pace'] else 500.0 # Default if no runs found
    return {"current_avg_pace": round(pace, 2)}

def analyze_drift_node(state: GraphState):
    """Gemini logic to determine if the user is drifting from the 60m goal."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    target = 360.0 # 6:00 min/km
    current = state["current_avg_pace"]
    diff = current - target
    
    prompt = f"""
    User Goal: 10k in 60 mins (360s/km).
    Current Week Average Pace: {current}s/km.
    Difference: {diff}s/km.
    
    Analyze the drift. If the difference is > 60s, suggest focus on aerobic base.
    If difference is < 30s, suggest speed intervals.
    Return a concise status and one training adjustment.
    """
    
    response = llm.invoke(prompt)
    return {"drift_status": "Off Track" if diff > 20 else "On Track", 
            "suggested_adjustments": response.content}

def update_plan_node(state: GraphState):
    """Saves the status back to the DB."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute(
        "UPDATE training_plans SET current_status = %s, last_updated = NOW() WHERE goal_name = '10k_60mins'",
        (state["suggested_adjustments"],)
    )
    conn.commit()
    conn.close()
    print("--- Plan Updated in Database ---")
    return state

# --- Build the Graph ---

workflow = StateGraph(GraphState)

workflow.add_node("fetch_performance", fetch_performance_node)
workflow.add_node("analyze_drift", analyze_drift_node)
workflow.add_node("update_plan", update_plan_node)

workflow.set_entry_point("fetch_performance")
workflow.add_edge("fetch_performance", "analyze_drift")
workflow.add_edge("analyze_drift", "update_plan")
workflow.add_edge("update_plan", END)

app = workflow.compile()