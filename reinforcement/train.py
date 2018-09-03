from reinforcement.agent import Agent
import sys
from reinforcement.functions import *
import sys

stock_name = "AAL"
window_size = 10
episode_count = 1

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    state = getState(data, 10, window_size + 1)

    total_profit = 0
    agent.inventory = []
    le = 0

    for t in range(l):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 1

        if action == 1:
            agent.inventory.append(data[t])
            print("Buy: " + str(data[t]))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(get_change(bought_price, data[t]), 1)
            total_profit += data[t] - bought_price
            print("Sell: " + str(get_change(bought_price, data[t])))

        elif action == 0 and len(agent.inventory) > 0:
            reward = max(get_change(data[t], data[t + 1]), 1)
            print("Hold: " + str(get_change(data[t], data[t + 1])))

        done = True if t == l - 5 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

        if le < sum(agent.inventory):
            le = sum(agent.inventory)

        if done:
            print("///////////////////////////////////////////////")
            print("Leftover shares: " + str(len(agent.inventory)))
            for i in range(len(agent.inventory)):
                bought_price = agent.inventory.pop(0)
                total_profit += data[t] - bought_price
            print("Total $ needed: " + str(le))
            print("Total Profit ($): " + str(total_profit))
            print("Total Profit (%): " + str(get_change(le, le + total_profit) * 100))
            print("///////////////////////////////////////////////")
            break

