class FeaturesHelper:
    def __init__(self):
        self.__features_name = [
            'vehicle_model', 'fuel_type', 'transmission_type', 'maximum_power(hp)',
            'maximum_torque(N/m)', 'maximum_speed (km/h)', 'steering_radius(m)',
            'vehicle_acceleration(0-100 km/h in seconds)', 'vehicle_mileage(mpg)',
            'vehicle_sprung_mass(kg)', 'tyre_camber_angle(degree)', 'tyre_brand',
            'tyre_size', 'tread_material', 'Standard_tread_depth(mm)', 'tread_pattern',
            'country', 'tread_wear_rating (UTQG)', 'average_tread_temperature(celsius)',
            'recommended_inflation_pressure(psi)', 'average_inflation_pressure(psi)',
            'tyre_age(years)', 'number_of_punctures', 'current_tread_depth(mm)',
            'road_condition', 'weather_condition', 'axle_type(driven/dead)',
            'expected_tyre_life(km)', 'retreaded', 'kilometers_driven(km)'
        ]
        self.__target_name = 'remaining_useful_life(km)'

    def get_features(self): return self.__features_name
    def get_target(self): return self.__target_name
