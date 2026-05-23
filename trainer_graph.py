# --- Workout Plan Generation ---
def generate_weekly_workout_plan(activity_types, workouts):
    """
    Generate a simple week-long workout plan using available activity types and recent workouts.
    This is a placeholder logic. You can customize it further as needed.
    """
    # Map activity type IDs to names
    type_map = {a['id']: a['name'] for a in activity_types}

    # Example: alternate running, strength, and rest days
    plan = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Pick most common activities in recent workouts
    from collections import Counter
    recent_types = [int(w['activity_type']) for w in workouts]
    most_common = [t for t, _ in Counter(recent_types).most_common(3)]
    # Fallback if not enough data
    if not most_common:
        most_common = [37, 20, 13]  # Running, Strength, Walking

    for i, day in enumerate(days):
        if i % 3 == 0:
            act = most_common[0]  # Running
        elif i % 3 == 1:
            act = most_common[1] if len(most_common) > 1 else most_common[0]
        else:
            act = most_common[2] if len(most_common) > 2 else most_common[0]
        plan.append(f"{day}: {type_map.get(act, 'Workout')}")
    return plan
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
    activity_types: list
    workouts: list
    splits: list
    stats: list

# --- Nodes ---

def fetch_performance_node(state: GraphState):
    """Fetch all relevant data from all tables for the last 7 days."""
    print(f"\n[Node: Fetching Data] Accessing database '{PG_DB}'...")
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1. Fetch all activity types
        cur.execute("SELECT * FROM workout_activity_types;")
        activity_types = cur.fetchall()
        print("Activity Types:")
        for row in activity_types:
            print(row)

        # 2. Fetch all workouts in the last 7 days
        cur.execute("""
            SELECT * FROM workouts
            WHERE start_date > CURRENT_DATE - INTERVAL '7 days'
        """)
        workouts = cur.fetchall()
        print("\nWorkouts (last 7 days):")
        for row in workouts:
            print(row)

        # 3. Fetch all splits for those workouts
        workout_ids = tuple([w['id'] for w in workouts])
        splits = []
        if workout_ids:
            cur.execute(f"""
                SELECT * FROM workout_splits
                WHERE workout_id IN %s
            """, (workout_ids,))
            splits = cur.fetchall()
        print("\nWorkout Splits (for last 7 days workouts):")
        for row in splits:
            print(row)

        # 4. Fetch all stats for those workouts
        stats = []
        if workout_ids:
            cur.execute(f"""
                SELECT * FROM workout_stats
                WHERE workout_id IN %s
            """, (workout_ids,))
            stats = cur.fetchall()
        print("\nWorkout Stats (for last 7 days workouts):")
        for row in stats:
            print(row)

        # Calculate average pace for running (activity_type 37)
        running_ids = [w['id'] for w in workouts if int(w['activity_type']) == 37]
        avg_pace = 490.0
        if running_ids:
            cur.execute(f"""
                SELECT AVG(avg_pace_seconds_per_km) as avg_pace FROM workout_splits WHERE workout_id IN %s
            """, (tuple(running_ids),))
            res = cur.fetchone()
            if res and res['avg_pace']:
                avg_pace = float(res['avg_pace'])
        print(f"\nResult: Average pace for running in last 7 days is {round(avg_pace, 2)} s/km.")

        conn.close()
        return {
            "current_avg_pace": round(avg_pace, 2),
            "activity_types": activity_types,
            "workouts": workouts,
            "splits": splits,
            "stats": stats,
            # Pass through other state fields if present
            "target_pace": state.get("target_pace", 360.0),
            "drift_status": state.get("drift_status", ""),
            "suggested_adjustments": state.get("suggested_adjustments", ""),
            "week_number": state.get("week_number", 1),
        }

    except Exception as e:
        conn.close()
        print(f"Error executing query: {e}")
        # Fallback to baseline so the graph doesn't crash
        return {
            "current_avg_pace": 490.0,
            "activity_types": [],
            "workouts": [],
            "splits": [],
            "stats": [],
            "target_pace": state.get("target_pace", 360.0),
            "drift_status": state.get("drift_status", ""),
            "suggested_adjustments": state.get("suggested_adjustments", ""),
            "week_number": state.get("week_number", 1),
        }

def analyze_drift_node(state: GraphState):
    """Gemini-powered analysis of the gap between current pace and 60m target."""
    print(f"[Node: Analyze Drift] Sending data to Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
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
    drift_status = "Losing Ground" if diff > 10 else "Gaining Momentum"
    
    print(f"Drift Analysis complete. Status: {drift_status}")
    return {
        "drift_status": drift_status, 
        "suggested_adjustments": response.content
    }

def update_plan_node(state: GraphState):
    """Persist the adjustment and drift status back into the database."""
    print(f"[Node: Update Plan] Saving adjustments to 'training_plans' table...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Ensure the adjustment is a string. 
    # Sometimes LLM responses are objects; we cast to string to be safe.
    adjustment_text = str(state.get("suggested_adjustments", ""))
    
    query = """
        INSERT INTO training_plans (goal_name, current_status, last_updated)
        VALUES ('10k_60mins', %s, NOW())
        ON CONFLICT (goal_name) 
        DO UPDATE SET current_status = EXCLUDED.current_status, last_updated = NOW();
    """
    
    try:
        # 2. Use a tuple (adjustment_text,) for the parameters
        cur.execute(query, (adjustment_text,))
        conn.commit()
        print(f"Successfully updated database for goal: '10k_60mins'.")
    except Exception as e:
        print(f"Error updating database: {e}")
        conn.rollback()
    finally:
        conn.close()
        
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

# --- Execution ---

if __name__ == "__main__":
    print("--- Starting Weekly Training Audit ---")
    final_state = app.invoke({"week_number": 1})

    print("\n--- FINAL COACHING SUMMARY ---")
    print(f"Goal: 10k in 60 Minutes")
    print(f"Status: {final_state['drift_status']}")
    print(f"Advice: {final_state['suggested_adjustments']}")
    print("--------------------------------------")

    # Generate and print a week-long workout plan
    print("\n--- WEEKLY WORKOUT PLAN ---")
    plan = generate_weekly_workout_plan(final_state.get('activity_types', []), final_state.get('workouts', []))
    for day_plan in plan:
        print(day_plan)
    print("--------------------------------------")