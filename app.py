import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from viktor import ViktorController
from viktor.parametrization import ViktorParametrization, GeoPointField, Text, MapSelectInteraction, SetParamsButton
from viktor.parametrization import NumberField, Tab
from viktor.result import SetParamsResult
from viktor.views import MapPolygon, MapResult, MapPoint, MapView, MapLine, MapLabel, Color


class Predictor():
    def __init__(self, csv_file="data/kc_house_data.csv") -> None:
        # Cargar datos
        self.data = pd.read_csv(csv_file)
        
        # Definimos las etiquetas
        self.labels = self.data['price']

        # Creamos un DataFrame de entrenamiento removiendo
        # las columnas id y price
        self.train = self.data.drop(["id", "price", "date"], axis=1)

        # Generamos conjuntos de entrenamiento y de prueba
        x_train , x_test , y_train , y_test = train_test_split(self.train,
                                                               self.labels,
                                                               test_size = 0.10,
                                                               random_state =2)

        # Generamos un regresor utilizando ensamblajes y Gradient Bootsting
        self.regressor = ensemble.GradientBoostingRegressor(n_estimators = 400,
                                                            max_depth = 5,
                                                            min_samples_split = 2,
                                                            learning_rate = 0.1,
                                                            loss = 'squared_error')
        
        # Entrenamos el regresor
        self.regressor.fit(x_train, y_train)

    def predict(self, input):
        self.prediction = self.regressor.predict(input)
        return self.prediction

# Carga de datos de viviendas
houses = pd.read_csv("data/kc_house_data.csv")

predictor = Predictor(csv_file="data/kc_house_data.csv")

class Parametrization(ViktorParametrization):
    intro = Text("""
# 🏠 App para estimación de costo de vivienda

En esta app puedes realizar la estimación del costo de una vivienda utilizando inteligencia artificial.
                 
De esta manera de puede hacer un análisis financiero de propiedades inmoviliarias.

**Selecciona el punto** correpondiente a la ubicación de la vivienda
    """)

    point = GeoPointField("Ubicación de nueva vivienda:")
    bedrooms = NumberField("Número de habitaciones:", default=4)
    bathrooms = NumberField("Número de baños:", default=4)
    floors = NumberField("Número de niveles:", default=1.5)
    sqft_living = NumberField("Área de construcción [m2]: ", default=390)
    yr_built = NumberField("Año de construcción:", default=1990)

class Controller(ViktorController):
    label = "Estimar costo de venta"
    parametrization = Parametrization

    @MapView("Análsis de ubicación", duration_guess=1)
    def generate_map(self, params, **kwargs):
        house = houses.iloc[0]
        lat = house.lat
        long = house.long

        # Crear punto en mapa utilizando las coordenadas
        some_point = MapPoint(lat, long, description='01', identifier='01')

        features = []
        # Crear puntos desde datos
        for i in range(20):
            house = houses.iloc[i]
            lat = house.lat
            long = house.long
            description = f"Precio: ${house.price} - "
            description += f"Habitaciones: {house.bedrooms} - "
            description += f"Niveles: {house.floors}"

            point_i = MapPoint(lat, long, description=description, identifier=str(i))
            features.append(point_i)
        
        # Obtener punto desde parámetros de entrada y agregarlo 
        # a las características si existe
        if params.point:
            input_point = MapPoint.from_geo_point(params.point)
            input_lat = input_point.lat
            input_long = input_point.lon
                # point = GeoPointField("Ubicación de nueva vivienda:")
                # bedrooms = NumberField("Número de habitaciones:")
                # bathrooms = NumberField("Número de baños:")
                # floors = NumberField("Número de niveles:")
                # sqft_living = NumberField("Área de construcción [m2]: ")
                # yr_built = NumberField("Año de construcción:")
            input_data = [params.bedrooms,                  # bedrooms
                          params.bathrooms,                 # bathrooms
                          params.sqft_living*10.7639,       # sqft_living
                          params.sqft_living*10.7639*1.2,   # sqft_lot
                          params.floors,                    # floors
                          0,                                # waterfront
                          0,                                # view
                          5,                                # condition
                          7,                                # grade
                          params.sqft_living*10.7639*0.6,   # sqft_above
                          params.sqft_living*10.7639*0.4,   # 
                          params.yr_built,
                          0,
                          98117,
                          input_lat,
                          input_long,
                          params.sqft_living*10.7639*.6,
                          params.sqft_living*10.7639*2]
            price = predictor.predict([input_data])[0]
            print(price)
            prediction_point = MapPoint(input_lat,
                                        input_long,
                                        icon="cross",
                                        description=f"Precio estimado: ${price:.2f}")
            features.append(prediction_point)

        return MapResult(features)

