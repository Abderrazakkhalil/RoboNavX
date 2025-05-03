import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
from IPython.display import clear_output
from collections import deque
import copy

# Configuration des graines aléatoires pour la reproductibilité
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==== On garde la même définition de l'environnement ====
class NavigationEnv(gym.Env):
    def __init__(self, grid_size=20, obstacle_count=5, dynamic_obstacles=True):
        super(NavigationEnv, self).__init__()
        
        # Paramètres de l'environnement
        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.dynamic_obstacles = dynamic_obstacles
        self.max_steps = grid_size * 3
        self.sensor_range = 5  # Augmenté pour une meilleure perception
        self.move_obstacle_probability = 0.5# Réduit pour plus de stabilité
        
        # Définition cohérente de l'espace d'observation
        # [x, y, sin(dir), cos(dir), distance_to_goal, angle_to_goal, 8 capteurs]
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(14,), dtype=np.float32)
        
        # Actions: 0=avancer, 1=tourner à gauche, 2=tourner à droite, 3=arrêt
        self.action_space = spaces.Discrete(4)
        
        # Initialisation
        self.grid = None
        self.start = None
        self.goal = None
        self.position = None
        self.direction = None
        self.steps = 0
        self.path = None
        self.current_path_index = 0
        self.obstacles = []
        
        # Variables pour le rendu
        self.fig = None
        self.ax = None
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Création de la grille vide
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Définition du départ et de l'objectif
        self.start = (1, 1)
        self.goal = (self.grid_size - 2, self.grid_size - 2)
        
        # Placement des obstacles fixes
        self.obstacles = []
        for _ in range(self.obstacle_count):
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                pos = (x, y)
                
                # Vérifier que l'obstacle n'est pas au départ ou à l'arrivée 
                if pos != self.start and pos != self.goal :
                    self.obstacles.append(pos)
                    self.grid[pos] = 1  # 1 représente un obstacle
                    break
        
        # Planification du chemin avec A*
        self.path = self.a_star(self.start, self.goal)
        
        # Si aucun chemin n'est trouvé, réessayer avec moins d'obstacles
        if not self.path and self.obstacles:
            # Retirer quelques obstacles
            reduced_obstacles = self.obstacles[:-2] if len(self.obstacles) > 2 else []
            self.obstacles = reduced_obstacles
            self.grid = np.zeros((self.grid_size, self.grid_size))
            for obs in self.obstacles:
                self.grid[obs] = 1
            self.path = self.a_star(self.start, self.goal)
        
        self.current_path_index = 0
        
        # Position et orientation initiales
        self.position = self.start
        self.direction = 0  # Angle en radians (0 = droite, π/2 = haut)
        self.steps = 0
        
        # Retourner l'état initial
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Anciennes valeurs pour calculer les récompenses
        old_position = self.position
        old_distance_to_goal = self._distance_to(self.goal)
        old_path_index = self.current_path_index
        
        # Application de l'action
        if action == 0:  # Avancer
            dx = np.cos(self.direction)
            dy = np.sin(self.direction)
            new_x = int(round(self.position[0] + dx))
            new_y = int(round(self.position[1] + dy))
            new_pos = (new_x, new_y)
            
            # Vérifier si la nouvelle position est valide
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and new_pos not in self.obstacles:
                self.position = new_pos
        
        elif action == 1:  # Tourner à gauche
            self.direction = (self.direction + np.pi/4) % (2 * np.pi)
        
        elif action == 2:  # Tourner à droite
            self.direction = (self.direction - np.pi/4) % (2 * np.pi)
        
        # Action 3: Arrêt (rien à faire)
        
        # Mise à jour de l'index du chemin si on a progressé
        self._update_path_index()
         
        if self.dynamic_obstacles and random.random() < self.move_obstacle_probability:
           self._move_random_obstacle()
           # Replanifier le chemin si nécessaire
           if random.random() < 0.3:  # Occasionnellement replanifier
               self.path = self.a_star(self.position, self.goal)
               self.current_path_index = 0
               self._update_path_index()
        
        # Vérifier si on est arrivé à l'objectif
        done = self.position == self.goal
        
        # Vérifier si on a dépassé le nombre maximal de pas
        if self.steps >= self.max_steps:
            done = True
        
        # Calcul de la récompense
        reward = self._calculate_reward(old_position, old_distance_to_goal, old_path_index, action)
        
        # Nouvel état
        observation = self._get_observation()
        
        # Informations supplémentaires
        info = {
            'position': self.position,
            'goal': self.goal,
            'path': self.path,
            'current_index': self.current_path_index,
            'distance_to_goal': self._distance_to(self.goal)
        }
        
        truncated = False  # Pour la compatibilité avec Gymnasium
        
        return observation, reward, done, truncated, info
    
    def _update_path_index(self):
        """Met à jour l'index du chemin en fonction de la position actuelle."""
        if not self.path:
            return
            
        # Trouver le point du chemin le plus proche
        min_distance = float('inf')
        closest_index = self.current_path_index
        
        # Commencer à chercher à partir de l'index actuel
        for i in range(self.current_path_index, len(self.path)):
            point = self.path[i]
            dist = self._distance_to(point)
            if dist < min_distance:
                min_distance = dist
                closest_index = i
            if dist < 0.5:  # Si on est très proche d'un point du chemin
                break
        
        # Mettre à jour uniquement si on avance sur le chemin
        if closest_index > self.current_path_index:
            self.current_path_index = closest_index
    
    def _calculate_reward(self, old_position, old_distance_to_goal, old_path_index, action):
        """Système de récompense amélioré pour DQN."""
        reward = 0
        
        # Récompense basée sur la progression vers l'objectif
        current_distance_to_goal = self._distance_to(self.goal)
        distance_improvement = old_distance_to_goal - current_distance_to_goal
        reward += distance_improvement * 10  # Récompense proportionnelle au progrès
        
        # Récompense pour avancer sur le chemin
        if self.current_path_index > old_path_index:
            progress = self.current_path_index - old_path_index
            reward += progress * 5
        
        # Pénalité pour être immobile (sauf si on est en train de tourner)
        if self.position == old_position and action != 1 and action != 2:
            reward -= 0.5
        
        # Pénalité pour être sur un obstacle (defensive coding)
        if self.position in self.obstacles:
            reward -= 10
            
        # Pénalité légère pour chaque pas (encourage l'efficacité)
        reward -= 0.1
        
        # Récompense pour avoir atteint l'objectif
        if self.position == self.goal:
            reward += 100
            
        # Bonus pour rester proche du chemin optimal
        if self.path:
            dist_to_path = self._distance_to_path()
            path_alignment_reward = max(0, 1 - dist_to_path/2)
            reward += path_alignment_reward
        
        return reward
    
    def _distance_to(self, point):
        """Calcule la distance euclidienne entre la position actuelle et un point."""
        return np.sqrt((self.position[0] - point[0])**2 + (self.position[1] - point[1])**2)
    
    def _distance_to_path(self):
        """Calcule la distance minimale entre la position actuelle et le chemin planifié."""
        if not self.path:
            return 0
        
        # Trouver le point du chemin le plus proche
        min_distance = float('inf')
        start_index = max(0, self.current_path_index - 2)  # Considérer quelques points en arrière
        for point in self.path[start_index:]:
            dist = self._distance_to(point)
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def _get_observation(self):
        """Crée une observation plus riche et cohérente."""
        # Position normalisée
        norm_x = 2 * (self.position[0] / self.grid_size) - 1  # [-1, 1]
        norm_y = 2 * (self.position[1] / self.grid_size) - 1  # [-1, 1]
        
        # Direction (sinus et cosinus pour éviter les discontinuités)
        dir_cos = np.cos(self.direction)
        dir_sin = np.sin(self.direction)
        
        # Distance et direction vers l'objectif
        dx_goal = self.goal[0] - self.position[0]
        dy_goal = self.goal[1] - self.position[1]
        distance_to_goal = np.sqrt(dx_goal**2 + dy_goal**2) / self.grid_size  # Normalisé
        angle_to_goal = np.arctan2(dy_goal, dx_goal)
        # Angle relatif entre la direction du robot et la direction vers l'objectif
        angle_diff_goal = (angle_to_goal - self.direction) % (2 * np.pi)
        if angle_diff_goal > np.pi:
            angle_diff_goal -= 2 * np.pi
        norm_angle_diff_goal = angle_diff_goal / np.pi  # [-1, 1]
        
        # Distance et direction vers le prochain point du chemin
        if self.path and self.current_path_index < len(self.path) - 1:
            next_point = self.path[self.current_path_index + 1]
            dx_path = next_point[0] - self.position[0]
            dy_path = next_point[1] - self.position[1]
            distance_to_next = np.sqrt(dx_path**2 + dy_path**2) / self.grid_size
            angle_to_next = np.arctan2(dy_path, dx_path)
            angle_diff_path = (angle_to_next - self.direction) % (2 * np.pi)
            if angle_diff_path > np.pi:
                angle_diff_path -= 2 * np.pi
            norm_angle_diff_path = angle_diff_path / np.pi
        else:
            distance_to_next = 0
            norm_angle_diff_path = 0
        
        # Capteurs pour détecter les obstacles (8 directions)
        sensor_data = self._get_sensor_data()
        
        # Assemblage de l'observation
        observation = np.array([
            norm_x, norm_y,
            dir_sin, dir_cos,
            distance_to_goal, norm_angle_diff_goal,
            distance_to_next, norm_angle_diff_path,
            *sensor_data
        ], dtype=np.float32)
        
        return observation
    
    def _get_sensor_data(self):
        """Simule des capteurs dans 8 directions pour détecter les obstacles."""
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        sensor_data = []
        
        for angle in angles:
            # Direction absolue du capteur
            sensor_angle = (self.direction + angle) % (2 * np.pi)
            dx = np.cos(sensor_angle)
            dy = np.sin(sensor_angle)
            
            # Vérifier la distance à l'obstacle le plus proche dans cette direction
            for distance in range(1, self.sensor_range + 1):
                x = int(round(self.position[0] + dx * distance))
                y = int(round(self.position[1] + dy * distance))
                
                # Si hors de la grille ou obstacle
                if (x < 0 or x >= self.grid_size or 
                    y < 0 or y >= self.grid_size or 
                    (x, y) in self.obstacles):
                    # Normaliser la distance (plus proche = plus grande valeur)
                    sensor_data.append((self.sensor_range - distance + 1) / self.sensor_range)
                    break
            else:
                # Aucun obstacle détecté
                sensor_data.append(0.0)
        
        return sensor_data
    
    def _move_random_obstacle(self):
        """Déplace un obstacle aléatoire de manière plus stratégique."""
        if not self.obstacles:
            return
            
        # Sélectionner un obstacle aléatoire mais pas trop proche du robot
        safe_obstacles = [o for o in self.obstacles if self._distance_to(o) > 2]
        if not safe_obstacles:
            return
        
        obstacle_idx = self.obstacles.index(random.choice(safe_obstacles))
        obstacle = self.obstacles[obstacle_idx]
        
        # Directions possibles
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        # Essayer de déplacer l'obstacle dans une direction valide
        for dx, dy in directions:
            new_x, new_y = obstacle[0] + dx, obstacle[1] + dy
            new_pos = (new_x, new_y)
            
            # Vérifier si la nouvelle position est valide
            if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and
                new_pos != self.position and new_pos != self.goal and
                new_pos not in self.obstacles):
                
                # Ne pas bloquer complètement le chemin vers l'objectif
                temp_obstacles = self.obstacles.copy()
                temp_obstacles[obstacle_idx] = new_pos
                temp_grid = np.zeros((self.grid_size, self.grid_size))
                for obs in temp_obstacles:
                    temp_grid[obs] = 1
                
                # Vérifier si un chemin existe toujours
                old_obstacles = self.obstacles.copy()
                self.obstacles = temp_obstacles
                temp_path = self.a_star(self.position, self.goal)
                self.obstacles = old_obstacles
                
                if temp_path:  # Seulement déplacer si un chemin existe toujours
                    # Mettre à jour la grille et la liste des obstacles
                    self.obstacles[obstacle_idx] = new_pos
                    self.grid[obstacle] = 0
                    self.grid[new_pos] = 1
                    break
    
    def a_star(self, start, goal):
        """Implémentation de l'algorithme A* pour la planification de chemin, avec timeout."""
        # Fonctions heuristiques (distance euclidienne pour plus de précision)
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        # Nœuds voisins accessibles
        def get_neighbors(node):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Ajouter des mouvements diagonaux
            result = []
            for dx, dy in directions:
                x, y = node[0] + dx, node[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size and (x, y) not in self.obstacles:
                    # Calculer le coût de déplacement (distance euclidienne)
                    cost = np.sqrt(dx**2 + dy**2)
                    result.append(((x, y), cost))
            return result
        
        # Initialisation des structures pour A*
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        # Protection contre les boucles infinies
        max_iterations = self.grid_size * self.grid_size * 2
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruire le chemin
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Inverse le chemin pour avoir départ->arrivée
            
            for neighbor, move_cost in get_neighbors(current):
                # Coût pour atteindre ce voisin depuis le départ
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Ce chemin vers le voisin est meilleur
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    # Ajouter à l'open set si pas déjà présent
                    if not any(neighbor == node for _, node in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Pas de chemin trouvé ou timeout
        return []
    
    def render(self):
        """Affiche l'environnement avec des améliorations visuelles."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
        
        self.ax.clear()
        
        # Afficher la grille avec un fond clair
        self.ax.set_facecolor('#f5f5f5')
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(np.arange(0, self.grid_size, 1))
        self.ax.set_yticks(np.arange(0, self.grid_size, 1))
        self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Afficher les obstacles
        for obs in self.obstacles:
            self.ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                           color='#d62728', alpha=0.8))
        
        # Afficher le départ et l'objectif
        self.ax.add_patch(plt.Circle((self.start[0], self.start[1]), 0.4, 
                                    color='#2ca02c', alpha=0.8, label='Départ'))
        self.ax.add_patch(plt.Circle((self.goal[0], self.goal[1]), 0.4, 
                                    color='#1f77b4', alpha=0.8, label='Objectif'))
        
        # Afficher le chemin planifié
        if self.path:
            path_x = [point[0] for point in self.path]
            path_y = [point[1] for point in self.path]
            self.ax.plot(path_x, path_y, 'y--', linewidth=2, alpha=0.7)
            
            # Marquer le point actuel du chemin
            if self.current_path_index < len(self.path):
                current_path_point = self.path[self.current_path_index]
                self.ax.add_patch(plt.Circle((current_path_point[0], current_path_point[1]), 
                                           0.2, color='yellow', alpha=0.8))
        
        # Afficher la position et l'orientation actuelles du robot
        self.ax.add_patch(plt.Circle((self.position[0], self.position[1]), 0.5, 
                                    color='#9467bd', alpha=0.8, label='Robot'))
        # Afficher la direction
        arrow_length = 0.8
        dx = arrow_length * np.cos(self.direction)
        dy = arrow_length * np.sin(self.direction)
        self.ax.arrow(self.position[0], self.position[1], dx, dy, 
                     head_width=0.3, head_length=0.3, fc='black', ec='black', alpha=0.8)
        
        # Afficher les capteurs
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        sensor_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        for i, angle in enumerate(angles):
            sensor_angle = (self.direction + angle) % (2 * np.pi)
            dx = np.cos(sensor_angle)
            dy = np.sin(sensor_angle)
            sensor_color = sensor_colors[i % len(sensor_colors)]
            
            # Détection d'obstacle
            for distance in range(1, self.sensor_range + 1):
                x = int(round(self.position[0] + dx * distance))
                y = int(round(self.position[1] + dy * distance))
                
                # Vérifier si on a trouvé un obstacle
                if (x < 0 or x >= self.grid_size or 
                    y < 0 or y >= self.grid_size or 
                    (x, y) in self.obstacles):
                    # Tracer jusqu'à l'obstacle
                    self.ax.plot([self.position[0], self.position[0] + dx * (distance - 0.5)],
                               [self.position[1], self.position[1] + dy * (distance - 0.5)],
                               '-', color=sensor_color, alpha=0.6, linewidth=2)
                    # Marquer le point de collision
                    self.ax.plot(self.position[0] + dx * (distance - 0.5),
                               self.position[1] + dy * (distance - 0.5),
                               'o', color=sensor_color, alpha=0.8, markersize=6)
                    break
            else:
                # Tracer tout le capteur si pas d'obstacle
                self.ax.plot([self.position[0], self.position[0] + dx * self.sensor_range],
                           [self.position[1], self.position[1] + dy * self.sensor_range],
                           '-', color=sensor_color, alpha=0.4, linewidth=1.5)
        
        # Ajouter une légende et des informations
        plt.title(f'Navigation RL - Étape: {self.steps}/{self.max_steps}', fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        # Ajouter des infos sur le côté
        info_text = (
            f"Position: {self.position}\n"
            f"Direction: {self.direction:.2f} rad\n"
            f"Distance à l'objectif: {self._distance_to(self.goal):.2f}\n"
        )
        if self.path:
            info_text += f"Index chemin: {self.current_path_index}/{len(self.path)-1}\n"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10)
        
        plt.tight_layout()
        plt.pause(0.05)
    
    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

# ==== Définition du modèle DQN ====
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # Caractéristiques générales
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Attention pour les capteurs (les 8 dernières features)
        self.sensor_attention = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Sigmoid()
        )
        
        # Avantage et valeur (architecture Dueling DQN)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Assurer que x est au bon format batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Séparer les capteurs du reste de l'état
        sensors = x[:, -8:]  # Les 8 dernières features sont les capteurs
        
        # Appliquer l'attention aux capteurs
        sensor_weights = self.sensor_attention(sensors)
        weighted_sensors = sensors * sensor_weights
        
        # Recombiner l'état
        x = torch.cat([x[:, :-8], weighted_sensors], dim=1)
        
        # Extraction de caractéristiques
        features = self.feature_extractor(x)
        
        # Dueling DQN: séparer avantage et valeur
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combiner valeur et avantage pour obtenir les Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

# ==== Mémoire de replay pour DQN ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)
# ==== Agent DQN ====
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, buffer_size=100000, batch_size=64,
                 target_update_freq=1000):
        self.action_dim = action_dim
        self.gamma = gamma  # Facteur d'actualisation
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_step = 0
        
        # Paramètres pour l'exploration epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Réseaux de neurones: principal et cible
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Mode évaluation pour le réseau cible
        
        # Mémoire de replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Pour suivre les pertes
        self.losses = []
        
    def select_action(self, state, evaluation=False):
        """Sélectionne une action selon la politique epsilon-greedy."""
        if not evaluation and random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: meilleure action selon le modèle
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        """Décroissance de epsilon pour réduire l'exploration au fil du temps."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire de replay."""
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        """Effectue une étape d'optimisation sur un lot de transitions."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Pas assez d'échantillons
        
        # Échantillonner un lot de transitions
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir en tenseurs PyTorch
        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_states)
        done_batch = torch.FloatTensor(dones).unsqueeze(1)
        
        # Calculer les valeurs Q actuelles Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calculer les valeurs Q cibles max_a' Q'(s',a')
        with torch.no_grad():
            # Double DQN: sélection de l'action par le réseau principal
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Évaluation de l'action par le réseau cible
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            
            # Calcul de la cible: r + gamma * max_a' Q'(s',a') * (1 - done)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Calculer la perte (erreur quadratique moyenne)
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour éviter l'explosion des gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Mettre à jour le réseau cible périodiquement
        self.learning_step += 1
        if self.learning_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save(self, path):
        """Enregistre le modèle."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Charge un modèle enregistré."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

# ==== Fonction d'entraînement ====
def train(env, agent, num_episodes=500, max_t=1000, epsilon_min=0.01, 
          update_every=4, eval_interval=20, render_training=False):
    """
    Entraîne l'agent dans l'environnement.
    
    Paramètres:
    - env: Environnement d'entraînement
    - agent: Agent DQN
    - num_episodes: Nombre d'épisodes d'entraînement
    - max_t: Nombre maximal d'étapes par épisode
    - epsilon_min: Valeur minimale d'epsilon (non utilisé actuellement)
    - update_every: Fréquence de mise à jour du réseau
    - eval_interval: Fréquence d'évaluation de l'agent
    - render_training: Afficher l'environnement pendant l'entraînement
    """
    scores = []
    epsilons = []
    avg_scores = []
    best_avg_score = -np.inf
    step_count = 0
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            # Sélectionner une action
            action = agent.select_action(state)
            
            # Exécuter l'action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Stocker la transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Mettre à jour l'état
            state = next_state
            score += reward
            step_count += 1
            
            # Optimiser le modèle si le moment est venu
            if step_count % update_every == 0:
                agent.optimize_model()
            
            # Afficher l'environnement si demandé
            if render_training:
                env.render()
            
            if done or truncated:
                break
        
        # Mettre à jour epsilon
        agent.update_epsilon()
        
        # Enregistrer le score
        scores.append(score)
        epsilons.append(agent.epsilon)
        
        # Calculer le score moyen des 100 derniers épisodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Afficher les informations
        if episode % 10 == 0:
            print(f'Épisode {episode}/{num_episodes} | Score: {score:.2f} | Moyenne: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f}')
        
        # Évaluer l'agent périodiquement
        if episode % eval_interval == 0:
            eval_score = evaluate(env, agent, 5, render=False)
            print(f'Évaluation: Score moyen sur 5 épisodes: {eval_score:.2f}')
            
            # Sauvegarder le meilleur modèle
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save('best_dqn_agent.pth')
                print(f'Nouveau meilleur score moyen: {best_avg_score:.2f}')
    
    # Enregistrer le modèle final
    agent.save('final_dqn_agent.pth')
    
    return scores, avg_scores, epsilons

# ==== Fonction d'évaluation ====
def evaluate(env, agent, num_episodes=10, render=True):
    """
    Évalue l'agent sans exploration.
    
    Paramètres:
    - env: Environnement d'évaluation
    - agent: Agent DQN
    - num_episodes: Nombre d'épisodes d'évaluation
    - render: Afficher l'environnement pendant l'évaluation
    """
    scores = []
    success_count = 0
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            # Sélectionner la meilleure action (sans exploration)
            action = agent.select_action(state, evaluation=True)
            
            # Exécuter l'action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Mettre à jour l'état et le score
            state = next_state
            score += reward
            
            # Afficher l'environnement si demandé
            if render:
                env.render()
            
            # Vérifier si l'épisode est terminé
            if done or truncated:
                # Vérifier si l'agent a atteint l'objectif
                if info['position'] == info['goal']:
                    success_count += 1
                break
        
        scores.append(score)
        if render:
            print(f'Épisode d\'évaluation {episode}: Score = {score:.2f}')
    
    avg_score = np.mean(scores)
    success_rate = success_count / num_episodes * 100
    
    print(f'Évaluation sur {num_episodes} épisodes:')
    print(f'Score moyen: {avg_score:.2f}')
    print(f'Taux de réussite: {success_rate:.1f}%')
    
    return avg_score

# ==== Visualisation des résultats ====
def plot_results(scores, avg_scores, epsilons):
    """Affiche les graphiques des scores et d'epsilon."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Graphique des scores
    ax1.plot(scores, label='Score', alpha=0.6, color='blue')
    ax1.plot(range(len(avg_scores)), avg_scores, label='Score moyen (100 ep.)', color='red')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Score')
    ax1.set_title('Évolution du score pendant l\'entraînement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique d'epsilon
    ax2.plot(epsilons, label='Epsilon', color='green')
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Décroissance d\'epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png')
    plt.show()

# ==== Programme principal ====
if __name__ == "__main__":
    # Création de l'environnement
    env = NavigationEnv(grid_size=20, obstacle_count=50, dynamic_obstacles=True)
    
    # Vérification de la dimension de l'observation
    test_observation, _ = env.reset()
    state_dim = len(test_observation)  # Utiliser la longueur réelle de l'observation
    print(f"Dimension réelle de l'état: {state_dim}")
    
    # Paramètres du problème (vérifiés)
    action_dim = env.action_space.n  # Nombre d'actions possibles
    
    # Création de l'agent avec la dimension correcte
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # Mode d'exécution: 'train' ou 'eval'
    mode = 'train'  # Changer à 'eval' pour évaluer un modèle sauvegardé
    
    if mode == 'train':
        # Entraînement
        scores, avg_scores, epsilons = train(
            env=env,
            agent=agent,
            num_episodes=1200,
            max_t=1000,
            update_every=4,
            eval_interval=50,
            render_training=False  # Mettre à True pour voir l'entraînement
        )
        
        # Afficher les résultats
        plot_results(scores, avg_scores, epsilons)
        
        # Évaluation du modèle final
        print("\nÉvaluation du modèle final:")
        evaluate(env, agent, num_episodes=10, render=True)
        
    elif mode == 'eval':
        # Charger un modèle sauvegardé
        try:
            agent.load('best_dqn_agent.pth')
            print("Modèle chargé avec succès!")
        except:
            print("Aucun modèle trouvé. Veuillez d'abord entraîner l'agent.")
            exit()
        
        # Évaluation du modèle chargé
        evaluate(env, agent, num_episodes=10, render=True)
    
    # Fermer l'environnement
    env.close()