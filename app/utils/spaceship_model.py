import pickle
import numpy as np


class SpaceshipModel:
    def __init__(self, model_path):
        self.home_planet_mapping = {"Earth": 1, "Europa": 2, "Mars": 3}
        self.destination_mapping = {
            "TRAPPIST-1e": 1,
            "PSO J318.5-22": 2,
            "55 Cancri e": 3,
        }
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def map_inputs(self, home_planet, cryo_sleep, destination, age, vip, total_spend):
        """Map user inputs to model input format."""
        home_planet_value = self.home_planet_mapping[home_planet]
        destination_value = self.destination_mapping[destination]
        cryo_sleep_value = 1 if cryo_sleep == "Yes" else 0
        vip_value = 1 if vip == "Yes" else 0
        return np.array(
            [
                [
                    home_planet_value,
                    cryo_sleep_value,
                    destination_value,
                    age,
                    vip_value,
                    total_spend,
                ]
            ]
        )

    def predict(self, input_data):
        prediction = self.model.predict(input_data)[0]
        return "Transported" if prediction == 1 else "Not Transported"
