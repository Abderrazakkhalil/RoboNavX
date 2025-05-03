import time
import threading
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import json
import os
import warnings
from sklearn.exceptions import DataConversionWarning

# Ignorer certains avertissements pour l'affichage
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot_battery.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RobotBatteryManager")

class BatteryManager:
    """
    Système intelligent de gestion de batterie pour robot avec surveillance
    en temps réel, alertes, et adaptation dynamique de la consommation.
    """

    # Constantes pour les seuils de batterie
    CRITICAL_THRESHOLD = 15  # Seuil critique en pourcentage
    LOW_THRESHOLD = 30       # Seuil bas en pourcentage
    
    # États possibles du robot
    ACTIVITIES = ["idle", "moving", "processing", "scanning"]
    
    # Modes d'alimentation
    POWER_MODES = {
        "normal": {
            "cpu_freq": 100,        # Fréquence CPU en pourcentage 
            "sensor_sampling": 10,  # Taux d'échantillonnage des capteurs en Hz
            "motion_speed": 100     # Vitesse de déplacement en pourcentage
        },
        "eco": {
            "cpu_freq": 60,
            "sensor_sampling": 5,
            "motion_speed": 70
        },
        "alert": {
            "cpu_freq": 30,
            "sensor_sampling": 2,
            "motion_speed": 40
        }
    }
    
    def __init__(self, model_path=None):
        """Initialise le gestionnaire de batterie."""
        # Paramètres configurables
        self.config = self._load_config()
        
        # État de la batterie
        self.battery_level = 100.0  # Pourcentage initial
        self.charging = False
        self.discharge_rate = 0.05  # Taux de décharge par défaut (% par seconde)
        
        # État du robot
        self.current_mode = "normal"
        self.robot_status = {
            "activity": "idle",
            "motion_level": 0,       # 0-100
            "ambient_temperature": 22,  # Celsius
            "system_load": 10         # Pourcentage
        }
        
        # Historique des données
        self.history_length = 100
        self.battery_history = deque(maxlen=self.history_length)
        self.status_history = deque(maxlen=self.history_length)
        
        # Modèle prédictif
        self.model = None
        self.load_model(model_path)
        
        # Surveillance en arrière-plan
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info("Système de gestion de batterie initialisé.")
    
    def _load_config(self):
        """Charge la configuration depuis un fichier ou utilise les valeurs par défaut"""
        default_config = {
            "critical_threshold": self.CRITICAL_THRESHOLD, 
            "low_threshold": self.LOW_THRESHOLD,
            "power_modes": self.POWER_MODES,
            "check_interval": 1.0,  # Intervalle de vérification en secondes
            "enable_predictions": True,
            "adaptive_thresholds": True
        }
        
        try:
            if os.path.exists("battery_config.json"):
                with open("battery_config.json", "r") as f:
                    config = json.load(f)
                    # Mettre à jour les valeurs par défaut avec celles du fichier
                    default_config.update(config)
                logger.info("Configuration chargée depuis battery_config.json")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de la configuration: {e}")
        
        return default_config
    
    def load_model(self, model_path):
        """Charge un modèle pré-entraîné ou crée un nouveau modèle."""
        if model_path and os.path.exists(model_path):
            try:
                import joblib
                self.model = joblib.load(model_path)
                logger.info(f"Modèle chargé depuis {model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {e}")
                self._create_default_model()
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Crée un modèle par défaut basé sur des règles simples"""
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        logger.info("Modèle prédictif par défaut créé")
        
        # Entraînement avec des données synthétiques
        X_train, y_train = self._generate_synthetic_data(1000)
        self.model.fit(X_train, y_train)
        logger.info("Modèle entraîné avec des données synthétiques")
    
    def _generate_synthetic_data(self, n_samples):
        """Génère des données synthétiques pour l'entraînement initial du modèle"""
        # Caractéristiques: [niveau_batterie, température, charge_système, niveau_mouvement]
        X = np.zeros((n_samples, 4))
        y = np.zeros(n_samples, dtype=str)
        
        for i in range(n_samples):
            # Génération aléatoire des caractéristiques
            battery = np.random.uniform(0, 100)
            temp = np.random.uniform(10, 40)
            load = np.random.uniform(0, 100)
            motion = np.random.uniform(0, 100)
            
            X[i] = [battery, temp, load, motion]
            
            # Règles logiques pour les étiquettes
            if battery < self.config["critical_threshold"]:
                y[i] = "alert"
            elif battery < self.config["low_threshold"] or (battery < 50 and load > 70):
                y[i] = "eco"
            else:
                y[i] = "normal"
        
        return X, y
    
    def start_monitoring(self):
        """Démarre la surveillance de la batterie en arrière-plan"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Surveillance de la batterie démarrée")
    
    def stop_monitoring(self):
        """Arrête la surveillance de la batterie"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Surveillance de la batterie arrêtée")
    
    def _monitoring_loop(self):
        """Boucle principale de surveillance exécutée en arrière-plan"""
        while self.monitoring:
            # Simuler la décharge de la batterie
            self._update_battery_level()
            
            # Collecte des données
            self._collect_data()
            
            # Analyse et prise de décision
            self._analyze_and_adapt()
            
            time.sleep(self.config["check_interval"])
    
    def _update_battery_level(self):
        """Met à jour le niveau de batterie en simulant la charge ou la décharge"""
        if self.charging:
            # Simulation de charge (plus lente près de 100%)
            charge_rate = 0.1 * (1 - (self.battery_level / 100) * 0.5)
            self.battery_level = min(100, self.battery_level + charge_rate)
        else:
            # Ajustement du taux de décharge selon le mode actuel
            mode_factor = {
                "normal": 1.0,
                "eco": 0.5,
                "alert": 0.25
            }.get(self.current_mode, 1.0)
            
            # Ajustement selon l'activité du robot
            activity_factor = {
                "idle": 0.3,
                "moving": 1.0,
                "processing": 0.8,
                "scanning": 0.7
            }.get(self.robot_status["activity"], 1.0)
            
            # Facteur de température (plus élevé = décharge plus rapide)
            temp_factor = 1.0 + max(0, (self.robot_status["ambient_temperature"] - 25) / 50)
            
            # Calcul du taux de décharge effectif
            effective_rate = self.discharge_rate * mode_factor * activity_factor * temp_factor
            
            # Mise à jour du niveau de batterie
            self.battery_level = max(0, self.battery_level - effective_rate)
    
    def _collect_data(self):
        """Collecte des données pour l'analyse et l'historique"""
        timestamp = time.time()
        
        # Enregistrement des données de batterie
        battery_data = {
            "timestamp": timestamp,
            "level": self.battery_level,
            "charging": self.charging,
            "mode": self.current_mode
        }
        self.battery_history.append(battery_data)
        
        # Enregistrement des données du robot
        status_data = self.robot_status.copy()
        status_data["timestamp"] = timestamp
        self.status_history.append(status_data)
    
    def _analyze_and_adapt(self):
        """Analyse l'état actuel et adapte le mode d'alimentation"""
        # Vérification des seuils critiques
        if self.battery_level <= self.config["critical_threshold"]:
            if self.current_mode != "alert":
                self._set_power_mode("alert")
                self._trigger_alert("CRITIQUE: Niveau de batterie critique ({:.1f}%)".format(self.battery_level))
        
        # Vérification des seuils bas
        elif self.battery_level <= self.config["low_threshold"]:
            if self.current_mode != "eco" and self.current_mode != "alert":
                self._set_power_mode("eco")
                self._trigger_alert("ALERTE: Niveau de batterie bas ({:.1f}%)".format(self.battery_level))
        
        # Vérification des conditions environnementales extrêmes
        elif self.robot_status["ambient_temperature"] > 35 and self.battery_level < 70:
            # Température élevée et batterie non pleine -> mode éco
            if self.current_mode == "normal":
                self._set_power_mode("eco")
                self._trigger_alert("ALERTE: Température élevée, passage en mode économie d'énergie")
        
        # Utilisation du modèle prédictif pour les cas non critiques
        elif self.config["enable_predictions"]:
            predicted_mode = self._predict_optimal_mode()
            if predicted_mode != self.current_mode:
                self._set_power_mode(predicted_mode)
    
    def _predict_optimal_mode(self):
        """Utilise le modèle prédictif pour déterminer le mode optimal"""
        if not self.model:
            return self.current_mode
        
        # Préparation des caractéristiques pour la prédiction
        features = np.array([
            self.battery_level,
            self.robot_status["ambient_temperature"],
            self.robot_status["system_load"],
            self.robot_status["motion_level"]
        ]).reshape(1, -1)
        
        try:
            # Prédiction du mode optimal
            predicted_mode = self.model.predict(features)[0]
            
            # Si le mode actuel est "alert" et la batterie est critique,
            # on maintient ce mode quelle que soit la prédiction
            if self.current_mode == "alert" and self.battery_level <= self.config["critical_threshold"]:
                return "alert"
            
            return predicted_mode
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return self.current_mode
    
    def _set_power_mode(self, mode):
        """Change le mode d'alimentation du robot."""
        if mode not in self.config["power_modes"]:
            logger.warning(f"Mode d'alimentation inconnu: {mode}")
            return False
        
        # Si le mode ne change pas, ne rien faire
        if mode == self.current_mode:
            return True
        
        # Appliquer les paramètres du nouveau mode
        settings = self.config["power_modes"][mode]
        
        # Pour cette simulation, on se contente de changer le mode
        old_mode = self.current_mode
        self.current_mode = mode
        
        logger.info(f"Mode d'alimentation changé: {old_mode} -> {mode}")
        
        return True
    
    def _trigger_alert(self, message):
        """Déclenche une alerte avec le message spécifié."""
        logger.warning(message)
    
    def update_robot_status(self, status_updates):
        """Met à jour l'état du robot."""
        self.robot_status.update(status_updates)
        logger.debug(f"État du robot mis à jour: {self.robot_status}")
    
    def set_charging(self, charging_state):
        """Définit l'état de charge de la batterie."""
        self.charging = charging_state
        logger.info(f"État de charge modifié: {'en charge' if charging_state else 'sur batterie'}")
    
    def set_battery_level(self, level):
        """Force le niveau de batterie à une valeur spécifique (pour les tests)."""
        if 0 <= level <= 100:
            self.battery_level = level
            logger.info(f"Niveau de batterie défini manuellement à {level}%")
        else:
            logger.warning(f"Niveau de batterie invalide: {level}")
    
    def get_battery_info(self):
        """Renvoie des informations complètes sur l'état de la batterie."""
        # Calcul de la tendance (pente de la décharge)
        trend = 0
        if len(self.battery_history) >= 2:
            recent_history = list(self.battery_history)[-10:]
            if len(recent_history) >= 2:
                levels = [entry["level"] for entry in recent_history]
                trend = (levels[-1] - levels[0]) / len(levels)
        
        # Estimation du temps restant
        remaining_time = float('inf') if trend >= 0 else -self.battery_level / trend
        
        return {
            "level": self.battery_level,
            "charging": self.charging,
            "mode": self.current_mode,
            "discharge_rate": self.discharge_rate,
            "trend": trend,
            "estimated_remaining_time": remaining_time,
            "critical_threshold": self.config["critical_threshold"],
            "low_threshold": self.config["low_threshold"]
        }
    
    def get_recommendations(self):
        """Génère des recommandations pour optimiser la durée de vie de la batterie."""
        recommendations = []
        
        # Analyse de la température
        if self.robot_status["ambient_temperature"] > 30:
            recommendations.append("Température ambiante élevée: envisager de déplacer le robot vers un environnement plus frais")
        
        # Analyse de la charge système
        if self.robot_status["system_load"] > 70:
            recommendations.append("Charge système élevée: considérer la fermeture de processus non essentiels")
        
        # Analyse du mouvement
        if self.robot_status["motion_level"] > 80 and self.battery_level < 50:
            recommendations.append("Niveau de mouvement élevé: réduire les déplacements pour économiser de l'énergie")
        
        return recommendations
    
    def train_model(self, X=None, y=None):
        """Entraîne ou ré-entraîne le modèle prédictif."""
        if X is None or y is None:
            # Utiliser les données historiques collectées
            if len(self.battery_history) < 50:
                logger.warning("Pas assez de données pour entraîner le modèle")
                # Complémenter avec des données synthétiques si l'historique est insuffisant
                X_synth, y_synth = self._generate_synthetic_data(500)
                X, y = X_synth, y_synth
            else:
                # Préparation des données à partir de l'historique
                X, y = self._prepare_training_data_from_history()
        
        # Entraînement du modèle
        if not self.model:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X, y)
        accuracy = self.model.score(X, y)
        
        logger.info(f"Modèle entraîné avec une précision de {accuracy:.2f}")
        return accuracy
    
    def _prepare_training_data_from_history(self):
        """Prépare les données d'entraînement à partir de l'historique"""
        X = []
        y = []
        
        # S'assurer que les historiques ont la même longueur
        min_length = min(len(self.battery_history), len(self.status_history))
        
        for i in range(min_length):
            battery_data = list(self.battery_history)[i]
            status_data = list(self.status_history)[i]
            
            # Caractéristiques
            features = [
                battery_data["level"],
                status_data["ambient_temperature"],
                status_data["system_load"],
                status_data["motion_level"]
            ]
            
            X.append(features)
            y.append(battery_data["mode"])
        
        return np.array(X), np.array(y)
    
    def save_model(self, filepath="battery_model.joblib"):
        """Sauvegarde le modèle entraîné."""
        if not self.model:
            logger.warning("Pas de modèle à sauvegarder")
            return False
            
        try:
            import joblib
            joblib.dump(self.model, filepath)
            logger.info(f"Modèle sauvegardé dans {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False


class BatterySimulator:
    """
    Classe pour simuler différents scénarios de batterie et visualiser les résultats.
    """
    
    def __init__(self):
        # Configuration du style pour les graphiques
        plt.style.use('dark_background')
        
        # Parseur d'arguments pour l'utilisation en ligne de commande
        self.parser = argparse.ArgumentParser(description='Simulateur de batterie pour robot')
        self.parser.add_argument('--scenario', type=str, default='normal',
                        choices=['normal', 'extreme_temp', 'high_load', 'mixed', 'charging'],
                        help='Scénario de test à exécuter')
        self.parser.add_argument('--duration', type=int, default=60,
                        help='Durée de la simulation en secondes')
        self.parser.add_argument('--graph', action='store_true',
                        help='Afficher les graphiques après la simulation')
        self.parser.add_argument('--train', action='store_true',
                        help='Entraîner le modèle après la simulation')
        args, unknown = self.parser.parse_known_args()

        # Données pour les graphiques
        self.data = {
            'timestamps': [],
            'battery_levels': [],
            'modes': [],
            'temps': [],
            'loads': [],
            'activities': [],
            'power_consumption': []
        }
    
    def run(self, scenario='normal', duration=60, show_graph=True, train_model=False):
        """
        Exécute une simulation selon le scénario spécifié.
        
        Args:
            scenario: Le scénario à simuler
            duration: Durée de la simulation en secondes
            show_graph: Indique s'il faut afficher les graphiques
            train_model: Indique s'il faut entraîner le modèle
        """
        print(f"Démarrage de la simulation: {scenario} pour {duration} secondes")
        
        # Initialisation du gestionnaire de batterie
        battery_manager = BatteryManager()
        battery_manager.start_monitoring()
        
        # Réinitialiser les données de graphique
        for key in self.data:
            self.data[key] = []
        
        # Configuration du scénario
        self._setup_scenario(battery_manager, scenario)
        
        # Boucle principale de simulation
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                current_time = time.time() - start_time
                
                # Mise à jour des états selon le scénario
                self._update_scenario(battery_manager, scenario, current_time)
                
                # Collecte des données pour les graphiques
                self._collect_data(battery_manager, current_time)
                
                # Affichage périodique
                self._display_status(battery_manager, current_time)
                
                time.sleep(0.1)  # Pause pour ne pas surcharger le CPU
        
        except KeyboardInterrupt:
            print("\nSimulation interrompue par l'utilisateur")
        finally:
            # Arrêt propre
            battery_manager.stop_monitoring()
            
            # Affichage des statistiques
            self._display_statistics(battery_manager)
            
            # Entraînement optionnel du modèle
            if train_model:
                print("\nEntraînement du modèle avec les données de la simulation...")
                accuracy = battery_manager.train_model()
                if accuracy is not None:
                    print(f"Précision du modèle après entraînement: {accuracy:.2f}")
                    battery_manager.save_model("simulation_model.joblib")
            
            # Affichage des graphiques
            if show_graph:
                self._plot_results(scenario)
    
    def _setup_scenario(self, battery_manager, scenario):
        """Configure le gestionnaire de batterie selon le scénario."""
        if scenario == 'normal':
            # Scénario normal avec variations légères
            battery_manager.set_battery_level(80)
        elif scenario == 'extreme_temp':
            # Scénario de température extrême
            battery_manager.set_battery_level(70)
            battery_manager.update_robot_status({"ambient_temperature": 38})
        elif scenario == 'high_load':
            # Scénario de charge système élevée
            battery_manager.set_battery_level(60)
            battery_manager.update_robot_status({"system_load": 85, "activity": "processing"})
        elif scenario == 'mixed':
            # Scénario mixte avec variations
            battery_manager.set_battery_level(50)
        elif scenario == 'charging':
            # Scénario avec charge intermittente
            battery_manager.set_battery_level(30)
    
    def _update_scenario(self, battery_manager, scenario, current_time):
        """Met à jour les états du robot selon le scénario et le temps actuel."""
        if scenario == 'normal':
            if current_time % 15 < 5:  # Activité cyclique
                battery_manager.update_robot_status({
                    "activity": "moving",
                    "motion_level": random.randint(40, 60),
                    "system_load": random.randint(20, 40)
                })
            else:
                battery_manager.update_robot_status({
                    "activity": "idle",
                    "motion_level": 0,
                    "system_load": random.randint(5, 15)
                })
        
        elif scenario == 'extreme_temp':
            # Température fluctuante mais élevée
            new_temp = 35 + random.randint(-3, 5)
            battery_manager.update_robot_status({
                "ambient_temperature": new_temp,
                "activity": random.choice(["processing", "scanning"]),
                "system_load": random.randint(30, 70)
            })
        
        elif scenario == 'high_load':
            # Charge système constamment élevée
            battery_manager.update_robot_status({
                "system_load": random.randint(75, 95),
                "motion_level": random.randint(10, 30)
            })
        
        elif scenario == 'mixed':
            # Changements aléatoires d'activité et d'environnement
            if current_time % 10 < 0.5:  # Changement toutes les ~10 secondes
                activity = random.choice(battery_manager.ACTIVITIES)
                temp = random.randint(20, 40)
                load = random.randint(10, 90)
                motion = random.randint(0, 100) if activity == "moving" else random.randint(0, 20)
                
                battery_manager.update_robot_status({
                    "activity": activity,
                    "ambient_temperature": temp,
                    "system_load": load,
                    "motion_level": motion
                })
        
        elif scenario == 'charging':
            # Alternance entre charge et décharge
            if int(current_time) % 20 < 10:
                if not battery_manager.charging:
                    print(f"[{current_time:.1f}s] Connexion au chargeur")
                    battery_manager.set_charging(True)
            else:
                if battery_manager.charging:
                    print(f"[{current_time:.1f}s] Déconnexion du chargeur")
                    battery_manager.set_charging(False)
    
    def _collect_data(self, battery_manager, current_time):
        """Collecte des données pour les graphiques."""
        battery_info = battery_manager.get_battery_info()
        
        # Calcul approximatif de la consommation d'énergie
        power_consumption = self._calculate_power_consumption(battery_manager)
        
        # Ajout des données
        self.data['timestamps'].append(current_time)
        self.data['battery_levels'].append(battery_info["level"])
        self.data['modes'].append(battery_info["mode"])
        self.data['temps'].append(battery_manager.robot_status["ambient_temperature"])
        self.data['loads'].append(battery_manager.robot_status["system_load"])
        self.data['activities'].append(battery_manager.robot_status["activity"])
        self.data['power_consumption'].append(power_consumption)
    
    def _calculate_power_consumption(self, battery_manager):
        """Calcule une estimation de la consommation d'énergie."""
        base_consumption = 5.0  # Watts
        
        # Facteurs de consommation basés sur l'état
        activity_consumption = {
            "idle": 1.0,
            "moving": 3.0,
            "processing": 2.5,
            "scanning": 2.0
        }.get(battery_manager.robot_status["activity"], 1.0)
        
        # Ajout de la consommation liée à la charge système
        system_consumption = battery_manager.robot_status["system_load"] / 20.0
        
        # Ajout de la consommation liée au mouvement
        motion_consumption = battery_manager.robot_status["motion_level"] / 10.0
        
        # Facteur du mode d'alimentation
        mode_factor = {
            "normal": 1.0,
            "eco": 0.6,
            "alert": 0.3
        }.get(battery_manager.current_mode, 1.0)
        
        # Calcul de la consommation totale
        total_consumption = (base_consumption + activity_consumption + 
                             system_consumption + motion_consumption) * mode_factor
        
        return total_consumption
    
    def _display_status(self, battery_manager, current_time):
        """Affiche périodiquement l'état du robot."""
        if int(current_time) % 5 == 0 and current_time > 0 and current_time % 5 < 0.5:
            battery_info = battery_manager.get_battery_info()
            print(f"[{current_time:.1f}s] Batterie: {battery_info['level']:.1f}%, "
                  f"Mode: {battery_info['mode']}, "
                  f"Activité: {battery_manager.robot_status['activity']}, "
                  f"Temp: {battery_manager.robot_status['ambient_temperature']}°C, "
                  f"Charge: {battery_manager.robot_status['system_load']}%")
            
            # Afficher les recommandations
            recommendations = battery_manager.get_recommendations()
            if recommendations:
                print("  Recommandations:")
                for rec in recommendations:
                    print(f"  - {rec}")
    
    def _display_statistics(self, battery_manager):
        """Affiche les statistiques finales de la simulation."""
        print("\n=== Résultats de la simulation ===")
        final_info = battery_manager.get_battery_info()
        print(f"Niveau final de batterie: {final_info['level']:.1f}%")
        print(f"Mode final: {final_info['mode']}")
        
        # Statistiques des modes
        mode_stats = {}
        for mode in self.data['modes']:
            mode_stats[mode] = mode_stats.get(mode, 0) + 1
        
        print("\nRépartition des modes:")
        for mode, count in mode_stats.items():
            percentage = (count / len(self.data['modes'])) * 100
            print(f"  {mode}: {percentage:.1f}% du temps")
    
    def _plot_results(self, scenario):
       """Affiche des graphiques améliorés des résultats de la simulation."""
       # Création d'une figure avec plusieurs sous-graphiques
       fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
       fig.suptitle(f'Résultats de simulation du scénario: {scenario}', fontsize=16)
    
       # Graphique 1: Niveau de batterie et mode d'alimentation
       ax1 = axs[0]
       ax1.plot(self.data['timestamps'], self.data['battery_levels'], 'c-', linewidth=2, label='Niveau batterie')
       ax1.set_ylabel('Niveau batterie (%)')
       ax1.set_ylim(0, 100)
       ax1.grid(True, alpha=0.3)
    
       # Ajout des zones colorées pour les modes d'alimentation
       mode_colors = {'normal': 'green', 'eco': 'yellow', 'alert': 'red'}
       mode_changes = []
       last_mode = None
    
       for i, mode in enumerate(self.data['modes']):
          if mode != last_mode:
            mode_changes.append((i, mode))
            last_mode = mode
    
       for i in range(len(mode_changes)):
          start_idx = mode_changes[i][0]
          end_idx = len(self.data['timestamps']) if i == len(mode_changes) - 1 else mode_changes[i+1][0]
          mode = mode_changes[i][1]
          ax1.axvspan(self.data['timestamps'][start_idx], 
                   self.data['timestamps'][end_idx-1] if end_idx < len(self.data['timestamps']) else self.data['timestamps'][-1], 
                   alpha=0.2, 
                   color=mode_colors.get(mode, 'gray'))
    
       # Légende des modes
       for mode, color in mode_colors.items():
           ax1.plot([], [], 's', color=color, alpha=0.2, label=f'Mode {mode}')
    
       ax1.legend(loc='upper right')
       ax1.set_title('Évolution du niveau de batterie et modes d\'alimentation')
    
       # Graphique 2: Température et charge système
       ax2 = axs[1]
       ax2.plot(self.data['timestamps'], self.data['temps'], 'r-', label='Température (°C)')
       ax2.set_ylabel('Température (°C)', color='r')
       ax2.tick_params(axis='y', labelcolor='r')
       ax2.grid(True, alpha=0.3)
    
       ax2_twin = ax2.twinx()
       ax2_twin.plot(self.data['timestamps'], self.data['loads'], 'b-', label='Charge système (%)')
       ax2_twin.set_ylabel('Charge système (%)', color='b')
       ax2_twin.tick_params(axis='y', labelcolor='b')
    
       ax2.set_title('Conditions environnementales et charge système')
    
       # Combinaison des légendes pour les deux y-axes
       lines1, labels1 = ax2.get_legend_handles_labels()
       lines2, labels2 = ax2_twin.get_legend_handles_labels()
       ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
       # Graphique 3: Consommation d'énergie et activités
       ax3 = axs[2]
       ax3.plot(self.data['timestamps'], self.data['power_consumption'], 'g-', linewidth=2, label='Consommation (W)')
       ax3.set_ylabel('Consommation (W)')
       ax3.grid(True, alpha=0.3)
       ax3.set_xlabel('Temps (s)')
    
       # Représentation des activités
       activity_colors = {
          'idle': 'lightblue',
          'moving': 'orange',
          'processing': 'purple',
          'scanning': 'pink'
       }
    
       activity_changes = []
       last_activity = None
    
       for i, activity in enumerate(self.data['activities']):
          if activity != last_activity:
            activity_changes.append((i, activity))
            last_activity = activity
    
       for i in range(len(activity_changes)):
          start_idx = activity_changes[i][0]
          end_idx = len(self.data['timestamps']) if i == len(activity_changes) - 1 else activity_changes[i+1][0]
          activity = activity_changes[i][1]
          ax3.axvspan(self.data['timestamps'][start_idx], 
                   self.data['timestamps'][end_idx-1] if end_idx < len(self.data['timestamps']) else self.data['timestamps'][-1], 
                   alpha=0.2, 
                   color=activity_colors.get(activity, 'gray'))
    
       # Légende des activités
       for activity, color in activity_colors.items():
          ax3.plot([], [], 's', color=color, alpha=0.2, label=f'Activité: {activity}')
    
       ax3.legend(loc='upper right')
       ax3.set_title('Consommation d\'énergie et activités du robot')
    
       # Ajustement de la mise en page
       plt.tight_layout()
       plt.subplots_adjust(top=0.9)
    
       # Sauvegarde et affichage
       plt.savefig(f"simulation_{scenario}.png", dpi=300, bbox_inches='tight')
       plt.show()


def main():
    """Fonction principale pour exécuter le simulateur avec des options par défaut."""
    simulator = BatterySimulator()
    
    # Ne pas utiliser les arguments de ligne de commande dans Colab
    # Utiliser des valeurs par défaut ou laisser l'utilisateur les spécifier
    
    # Vous pouvez ajuster ces valeurs selon vos besoins
    scenario ='mixed' # Options: 'normal', 'extreme_temp', 'high_load', 'mixed', 'charging'
    duration = 60  # Durée en secondes
    show_graph = True
    train_model = False
    
    simulator.run(
        scenario=scenario,
        duration=duration,
        show_graph=show_graph,
        train_model=train_model
    )


if __name__ == "__main__":
    main()