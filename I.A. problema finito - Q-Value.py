import numpy as np

gamma = 0.75
alpha = 0.9

locais = {'A': 0,
          'B': 1,
          'C': 2,
          'D': 3,
          'E': 4,
          'F': 5,
          'G': 6,
          'H': 7,
          'I': 8,
          'J': 9,
          'K': 10,
          'L': 11}

acoes = [0,1,2,3,4,5,6,7,8,9,10,11]

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

Q = np.array(np.zeros([12,12]))

for i in range(1000):
    estado_atual = np.random.randint(0,12)
    acoes_possiveis = []
    for j in range(12):
        if R[estado_atual, j] > 0:
            acoes_possiveis.append(j)
    proximo_estado = np.random.choice(acoes_possiveis)
    TD = R[estado_atual, proximo_estado] + gamma*Q[proximo_estado, np.argmax(Q[proximo_estado])] - Q[estado_atual, proximo_estado]        
    Q[estado_atual, proximo_estado] = Q[estado_atual, proximo_estado] + alpha*TD
    
print("Valores de Q: ")
print(Q.astype(int))
input()