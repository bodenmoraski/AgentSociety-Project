from agent import load_agent_from_json
from market import Market

def main():
    # Instantiate two agents from the same JSON config
    agent1 = load_agent_from_json("config/default.json")
    agent2 = load_agent_from_json("config/default.json")

    # Optionally, give agent2 a different id/name to distinguish
    agent2.id = 2
    agent2.name = "DefaultAgent2"

    # Create market with both agents
    market = Market([agent1, agent2])

    # Run a few simulation steps
    for step in range(5):
        print(f"\n--- Market Step {step+1} ---")
        market.step()
        print(f"Agent1 wealth: {agent1.wealth}, goods: {agent1.goods}")
        print(f"Agent2 wealth: {agent2.wealth}, goods: {agent2.goods}")

if __name__ == "__main__":
    main()