import numpy as np
import gym
import matplotlib.pyplot as plt
import gym_gridworld

# Q-Learning
def qlearning(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    rewards = []
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) #Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estañdo siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) #Calcula la nueva Q table.

            state = next_state

            if (MAX_STEPS-2)<actual_step:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}") 
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                print(f"Episode {episode} rewards: {rewards_epi}") 
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break  

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return rewards, Q

## SARSA
def sarsa(env, epsilon):
    rewards = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles
    
    Q = np.zeros((STATES, ACTIONS))
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)
            
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.
            	
            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * Q[next_state, action2] - Q[state, action]) #Calcula la nueva Q table.
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2)<actual_step:
                print (f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}") 
                if epsilon > 0.1: epsilon -= 0.0001

            if done:
                print (f"Episode {episode} rewards: {rewards_epi}") 
                rewards.append(rewards_epi) #Guarda las recompensas en una lista
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001
                break   

    print(Q)
    #print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return rewards, Q

## Double Q-Learning
def double_qlearning(env, epsilon):
    rewards = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles

    Q1 = np.zeros((STATES, ACTIONS)) #Q table 1
    Q2 = np.zeros((STATES, ACTIONS)) #Q table 2

    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()
        for actual_step in range(MAX_STEPS):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q1[state, :] + Q2[state, :])

            next_state, reward, done, _ = env.step(action)

            if np.random.uniform(0, 1) < 0.5:
                Q1[state, action] = Q1[state, action] + LEARNING_RATE * (reward + GAMMA * Q2[next_state, np.argmax(Q1[next_state, :])] - Q1[state, action])
            else:
                Q2[state, action] = Q2[state, action] + LEARNING_RATE * (reward + GAMMA * Q1[next_state, np.argmax(Q2[next_state, :])] - Q2[state, action])

            rewards_epi = rewards_epi + reward
            state = next_state

            if done:
                print(f"Episode {episode} rewards: {rewards_epi}")
                rewards.append(rewards_epi)
                break
    return rewards, Q1, Q2


#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    #pause=input()
    env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while True:
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: env.render()
            pause=input()
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    pause=input()
    env.close()
    print("Victorias: ", wins)

#Función para guardar los resultados
def save_results(rewards, Q, filename_rewards, filename_qvalues, plot_filename):
    np.savetxt(filename_rewards, rewards, fmt="%d")
    np.savetxt(filename_qvalues, Q, fmt="%d")

    plt.plot(rewards)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Recompensa v/s episodio")
    plt.savefig(plot_filename)
    plt.show()

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

#########################
####### Parte 1 a 3 #####
#########################
'''
#* Original Q-learning
EPISODES = 10000 
MAX_STEPS = 100 

LEARNING_RATE = 0.2  
GAMMA = 0.90 
epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

#Q = sarsa(env, epsilon)
rewards, Q = qlearning(env, epsilon)
save_results(rewards, Q, "mapa1Qlearning.txt", "mapa1Qvalues.txt", "map1QlearningPlot.png")
playgames(env, Q, 1, True)
env.close()

#* Original Sarsa
env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 100 

LEARNING_RATE = 0.2  
GAMMA = 0.90 
epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

#Q = sarsa(env, epsilon)
rewards, Q = qlearning(env, epsilon)
save_results(rewards, Q, "mapa1Sarsa.txt", "mapa1Sarsa.txt", "map1Sarsa1Plot.png")
env.close()

#* Mapa 2 Q-Learning

EPISODES = 10000 
MAX_STEPS = 300 

LEARNING_RATE = 0.3  
GAMMA = 0.95
epsilon = 1

print('Observation space\n')
print(env.observation_space)

print('Action space\n')
print(env.action_space)

#Q = sarsa(env, epsilon)
rewards, Q = qlearning(env, epsilon)
save_results(rewards, Q, "mapa2Qlearning.txt", "mapa2Qvalues.txt", "qlearningPlot.png")
playgames(env, Q, 1, True)
env.close()

#* Mapa 2 Q-Learning 2

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 300 

LEARNING_RATE = 0.1
GAMMA = 1
epsilon = 1

print('Observation space\n')
print(env.observation_space)

print('Action space\n')
print(env.action_space)

#Q = sarsa(env, epsilon)
rewards, Q = qlearning(env, epsilon)
save_results(rewards, Q, "mapa2Qlearning2.txt", "mapa2Qvalues2.txt", "qlearning2Plot.png")
playgames(env, Q, 1, True)
env.close()

#* Mapa 2 SARSA

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 200 

LEARNING_RATE = 0.01
GAMMA = 0.99
epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

rewards, Q = sarsa(env, epsilon)
save_results(rewards, Q, "mapa2SARSA.txt", "mapa2SARSAvalues.txt", "sarsaPlot.png")
playgames(env, Q, 1, True)
env.close()

#* Mapa 2 SARSA 2

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 200 

LEARNING_RATE = 0.1
GAMMA = 1
epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

rewards, Q = sarsa(env, epsilon)
save_results(rewards, Q, "mapa2SARSA2.txt", "mapa2SARSA2values.txt", "sarsa2Plot.png")
playgames(env, Q, 1, True)
env.close()


#* Mapa 2 double Q-Learning

EPISODES = 10000 
MAX_STEPS = 300 

LEARNING_RATE = 0.3  
GAMMA = 0.95
epsilon = 1

print('Observation space\n')
print(env.observation_space)

print('Action space\n')
print(env.action_space)

#Q = sarsa(env, epsilon)
rewards, Q1, Q2 = double_qlearning(env, epsilon)
save_results(rewards, Q1, "mapa2Qlearning.txt", "mapa2Qvalues.txt", "qlearningPlot.png")
playgames(env, Q1+Q2, 1, True)
env.close()

#* Mapa 2 Q-Learning 2

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 300 

LEARNING_RATE = 0.1
GAMMA = 1
epsilon = 1

print('Observation space\n')
print(env.observation_space)

print('Action space\n')
print(env.action_space)

rewards, Q1, Q2 = double_qlearning(env, epsilon)
save_results(rewards, Q1, "mapa2Qlearning2.txt", "mapa2Qvalues2.txt", "qlearning2Plot.png")
playgames(env, Q1+Q2, 1, True)
env.close()
'''


#* TD
'''
###########################
#### Implementacion TD ####
###########################

EPISODES = 10000
MAX_STEPS = 300

# Clase agente, recibe el numero de estados, el numero de acciones, el alpha, el gamma, el epsilon y el lambd, y tiene las funciones para escoger una accion, actualizar la Q table y actualizar el lambd
class Agent:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, lambd = 0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        self.Q = np.zeros((num_states, num_actions))
        self.E = np.zeros((num_states, num_actions))

    # Funcion para elegir una acción
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.Q[state, :])
        return action
        # Funcion para actualizar la tabla Q (Sarsa)
    def update_Q_sarsa(self, state, action, reward, next_state, next_action, done):
        delta = reward + self.gamma * self.Q[next_state, next_action] * (not done) - self.Q[state, action]
        self.E[state, action] += 1
        self.Q += self.alpha * delta * self.E
        self.E *= self.gamma * self.lambd


        #Funcion para actualizar la tabla Q (Q-learning)
    def update_Q_qlearning(self, state, action, reward, next_state, done):
        max_next_Q = np.max(self.Q[next_state, :])
        delta = reward + self.gamma * max_next_Q * (not done) - self.Q[state, action]
        self.E[state, action] += 1
        self.Q += self.alpha * delta * self.E
        self.E *= self.gamma * self.lambd

# Funcion para correr episodios
def run_episodes(agent, update_Q):
    rewards_per_episode = []
    for episode in range(EPISODES):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        for actual_step in range(MAX_STEPS):
            next_state, reward, done, info = env.step(action)
            next_action = agent.choose_action(next_state)
            if update_Q == agent.update_Q_qlearning:
                update_Q(state, action, reward, next_state, done)
            else:
                update_Q(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward
            if done:
                break
        rewards_per_episode.append(total_reward)
    return rewards_per_episode, agent.Q
num_states = env.observation_space.n
num_actions = env.action_space.n
agent = Agent(num_states, num_actions, alpha=0.1, gamma=0.95, epsilon=1.0)  

# Crear 2 agentes
agente_sarsa = Agent(num_states, num_actions, alpha=0.01, gamma=0.99, epsilon=1.0)
agente_qlearning = Agent(num_states, num_actions, alpha=0.1, gamma=0.95, epsilon=1.0)

# Episodios Sarsa
rewards_per_episode_sarsa, q_sarsa = run_episodes(agente_sarsa, agente_sarsa.update_Q_sarsa)

# Episodios Qlearning
rewards_per_episode_qlearning, q_qlearning = run_episodes(agente_qlearning, agente_qlearning.update_Q_qlearning)

# guardar datos
save_results(rewards_per_episode_sarsa, q_sarsa, "TDsarsa1.txt", "QTDsarsa1.txt", "TDsarsa1.png")
save_results(rewards_per_episode_qlearning, q_qlearning, "TDqlearning1.txt", "QTDlearning1.txt", "TDsarsa1.png")
'''