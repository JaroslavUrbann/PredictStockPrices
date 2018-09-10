from reinforcement.agent import Agent
import sys
from reinforcement.functions import *
import sys
import os.path

stock_name = "AAL"
window_size = 30
episode_count = 1

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 1

for e in range(episode_count):
    state = getState(data, 10, window_size + 1)

    total_profit = 0
    dollars_needed = 0
    posral_to = 0
    posral_to_holdem = 0

    for t in range(l):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 1

        if action == 1:
            agent.inventory.append(data[t])
            print("Buy: " + str(data[t]))

        if action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(get_change(bought_price, data[t]), 1)
            total_profit += data[t] - bought_price
            print("Sell: " + str(get_change(bought_price, data[t])))

        elif action == 2 and len(agent.inventory) == 0:
            print("posral to")
            posral_to += 1

        if action == 0 and len(agent.inventory) > 0:
            reward = max(get_change(data[t], data[t + 1]), 1)
            print("Hold: " + str(get_change(data[t], data[t + 1])))

        elif action == 0 and len(agent.inventory) == 0:
            print("posral to holdem")
            posral_to_holdem += 1

        done = True if t == l - 5 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) >= batch_size:
            agent.expReplay()

        if dollars_needed < sum(agent.inventory):
            dollars_needed = sum(agent.inventory)

        if done:
            agent.model.save('../models/reinforcement.h5')
            print("///////////////////////////////////////////////")
            print("Leftover shares: " + str(len(agent.inventory)))
            print("Posral to: " + str(posral_to))
            print("Posral to holdem: " + str(posral_to_holdem))
            for i in range(len(agent.inventory)):
                bought_price = agent.inventory.pop(0)
                total_profit += data[t] - bought_price
            print("Total $ needed: " + str(dollars_needed))
            print("Total Profit ($): " + str(total_profit))
            print("Total Profit (%): " + str(get_change(dollars_needed, dollars_needed + total_profit) * 100))
            print("///////////////////////////////////////////////")
            break

