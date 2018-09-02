from reinforcement.agent import Agent
import sys
from reinforcement.functions import *

stock_name = "AAL"
window_size = 10
episode_count = 1

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    state = getState(data, 0, window_size + 1)
    print(state)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(data[t])
            print("Buy: " + str(data[t]))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + str(data[t]) + " | Profit: " + str(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("Total Profit: " + str(total_profit))

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
